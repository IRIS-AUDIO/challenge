import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

import efficientnet.model as model
from swa import SWA
from pipeline import *
from transforms import *
from utils import *
from models import transformer_layer


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB4')
args.add_argument('--pretrain', type=bool, default=False)

# DATA
args.add_argument('--background_sounds', type=str,
                  default='/codes/generate_wavs/drone_normed_complex.pickle')
args.add_argument('--voices', type=str,
                  default='/codes/generate_wavs/voice_normed_complex.pickle')
args.add_argument('--labels', type=str,
                  default='/codes/generate_wavs/voice_labels_mfc.npy')
args.add_argument('--noises', type=str,
                  default='/codes/RDChallenge/tf_codes/sounds/noises_specs_2.pickle')

# TRAINING
args.add_argument('--optimizer', type=str, default='adam',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--lr_factor', type=float, default=0.7)
args.add_argument('--lr_patience', type=int, default=10)

args.add_argument('--epochs', type=int, default=200)
args.add_argument('--batch_size', type=int, default=16)
args.add_argument('--n_frame', type=int, default=1000)
args.add_argument('--steps_per_epoch', type=int, default=500)
args.add_argument('--l2', type=float, default=1e-6)

# AUGMENTATION
args.add_argument('--alpha', type=float, default=0.75)
args.add_argument('--snr', type=float, default=-10)
args.add_argument('--max_voices', type=int, default=4)
args.add_argument('--max_noises', type=int, default=2)


def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=16, n_mask=6) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=32) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    return specs, labels


def preprocess_labels(x, y):
    # preprocess y
    for i in range(5):
        y = tf.nn.max_pool1d(y, 2, strides=2, padding='SAME')
    return x, y


def load_data(path):
    if path.endswith('.pickle'):
        return pickle.load(open(path, 'rb'))
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('invalid file format')


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
    N_CLASSES = 30

    with strategy.scope():
        """ MODEL """
        x = tf.keras.layers.Input(shape=(257, None, 4))
        model = getattr(model, config.model)(
            include_top=False,
            weights=None,
            input_tensor=x,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )
        out = tf.transpose(model.output, perm=[0, 2, 1, 3])
        out = tf.keras.layers.Reshape([-1, out.shape[-1]*out.shape[-2]])(out)

        # Add Transformer Layer
        n_layers, n_dim, n_heads = 1, 256, 8
        out = tf.keras.layers.Dense(n_dim)(out)
        for i in range(n_layers):
            out = transformer_layer(n_dim, n_heads)(out)
        out = tf.keras.layers.Dense(N_CLASSES, activation='sigmoid')(out)
        model = tf.keras.models.Model(inputs=model.input, outputs=out)

        if config.optimizer == 'adam':
            opt = Adam(config.lr) 
        elif config.optimizer == 'sgd':
            opt = SGD(config.lr, momentum=0.9)
        else:
            opt = RMSprop(config.lr, momentum=0.9)

        if config.l2 > 0:
            model = apply_kernel_regularizer(
                model, tf.keras.regularizers.l2(config.l2))
        model.compile(optimizer=opt, 
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'AUC'])
        model.summary()
        
        if config.pretrain:
            model.load_weights(NAME)
            print('loaded pretrained model')

        """ DATA """
        # TRAINING DATA
        backgrounds = load_data(config.background_sounds)
        voices = load_data(config.voices)
        labels = load_data(config.labels)
        labels = np.eye(N_CLASSES, dtype='float32')[labels] # to one-hot vectors
        noises = load_data(config.noises)

        # PIPELINE
        pipeline = make_pipeline(backgrounds, 
                                 voices, labels,
                                 noises,
                                 n_frame=config.n_frame,
                                 max_voices=config.max_voices,
                                 max_noises=config.max_noises,
                                 n_classes=N_CLASSES,
                                 snr=config.snr)
        pipeline = pipeline.map(to_frame_labels, num_parallel_calls=AUTOTUNE)
        pipeline = pipeline.map(augment, num_parallel_calls=AUTOTUNE)
        pipeline = pipeline.batch(BATCH_SIZE, drop_remainder=False)
        # pipeline = pipeline.map(magphase_mixup(alpha=config.alpha, feat='complex'))
        pipeline = pipeline.map(minmax_norm_magphase)
        pipeline = pipeline.map(log_magphase)
        pipeline = pipeline.map(preprocess_labels)
        pipeline = pipeline.prefetch(AUTOTUNE)

        """ TRAINING """
        callbacks = [
            CSVLogger(NAME.replace('.h5', '.log'),
                      append=True),
            ReduceLROnPlateau(monitor='auc',
                              factor=config.lr_factor,
                              patience=config.lr_patience,
                              mode='max'),
            SWA(start_epoch=TOTAL_EPOCH//2, swa_freq=2),
            ModelCheckpoint(NAME,
                            monitor='auc',
                            mode='max',
                            save_best_only=True),
            TerminateOnNaN()
        ]

        model.fit(pipeline,
                  epochs=TOTAL_EPOCH,
                  batch_size=BATCH_SIZE,
                  steps_per_epoch=config.steps_per_epoch,
                  callbacks=callbacks)
