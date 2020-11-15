import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K

import efficientnet.model as model
from swa import SWA
from pipeline import *
from transforms import *
from utils import *
from models import transformer_layer


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--pretrain', type=bool, default=False)
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dim', type=int, default=256)
args.add_argument('--n_heads', type=int, default=8)

# DATA
# TRAINING
args.add_argument('--background_sounds', type=str,
                  default='/codes/generate_wavs/drone_normed_complex.pickle')
args.add_argument('--voices', type=str,
                  default='/codes/generate_wavs/voice_normed_complex.pickle')
args.add_argument('--labels', type=str,
                  default='/codes/generate_wavs/voice_labels_mfc.npy')
args.add_argument('--noises', type=str,
                  default='/codes/RDChallenge/tf_codes/sounds/noises_specs.pickle')
# TEST
args.add_argument('--test_background_sounds', type=str,
                  default='/codes/generate_wavs/test_drone_normed_complex.pickle')
args.add_argument('--test_voices', type=str,
                  default='/codes/generate_wavs/test_voice_normed_complex.pickle')
args.add_argument('--test_labels', type=str,
                  default='/codes/generate_wavs/test_voice_labels_mfc.npy')
args.add_argument('--n_mels', type=int, default=128)
args.add_argument('--n_classes', type=int, default=30)

# TRAINING
args.add_argument('--optimizer', type=str, default='adam',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--lr_factor', type=float, default=0.7)
args.add_argument('--lr_patience', type=int, default=10)

args.add_argument('--epochs', type=int, default=500)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--n_frame', type=int, default=2048)
args.add_argument('--steps_per_epoch', type=int, default=200)
args.add_argument('--l2', type=float, default=1e-6)

# AUGMENTATION
args.add_argument('--snr', type=float, default=-15)
args.add_argument('--max_voices', type=int, default=10)
args.add_argument('--max_noises', type=int, default=5)


def minmax_log_on_mel(mel, labels=None):
    axis = tuple(range(1, len(mel.shape)))

    # MIN-MAX
    mel_max = tf.math.reduce_max(mel, axis=axis, keepdims=True)
    mel_min = tf.math.reduce_min(mel, axis=axis, keepdims=True)
    mel = (mel-mel_min) / (mel_max-mel_min+EPSILON)

    # LOG
    mel = tf.math.log(mel + EPSILON)

    if labels is not None:
        return mel, labels
    return mel


def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=24, n_mask=8) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=32) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    return specs, labels


def preprocess_labels(x, y):
    # preprocess y
    for i in range(5):
        y = tf.nn.max_pool1d(y, 2, strides=2, padding='SAME')
    return x, y


def custom_scheduler(d_model, warmup_steps=4000):
    # https://www.tensorflow.org/tutorials/text/transformer#optimizer
    d_model = tf.cast(d_model, tf.float32)

    def _scheduler(step):
        step = tf.cast(step+1, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (warmup_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)
    return _scheduler


def focal_loss(y_true, y_pred, 
               alpha=0.75, # 0.25, 
               gamma=2.0):
    # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/sigmoid_focal_crossentropy
    # assume y_pred is prob

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred)

    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


def make_dataset(config, training=True):
    # Load required datasets
    if training:
        backgrounds = load_data(config.background_sounds)
        voices = load_data(config.voices)
        labels = load_data(config.labels)
    else:
        backgrounds = load_data(config.test_background_sounds)
        voices = load_data(config.test_voices)
        labels = load_data(config.test_labels)
    labels = np.eye(N_CLASSES, dtype='float32')[labels] # to one-hot vectors
    noises = load_data(config.noises)

    # Make pipeline and process the pipeline
    pipeline = make_pipeline(backgrounds, 
                             voices, labels,
                             noises,
                             n_frame=config.n_frame,
                             max_voices=config.max_voices,
                             max_noises=config.max_noises,
                             n_classes=config.n_classes,
                             snr=config.snr)
    pipeline = pipeline.map(to_frame_labels)
    if training:
        pipeline = pipeline.map(augment)
    pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    pipeline = pipeline.map(minmax_log_on_mel)
    pipeline = pipeline.map(preprocess_labels)
    return pipeline.prefetch(AUTOTUNE)


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
    N_CLASSES = config.n_classes

    with strategy.scope():
        """ MODEL """
        x = tf.keras.layers.Input(shape=(config.n_mels, None, 2))
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
        if config.n_layers > 0:
            out = tf.keras.layers.Dense(config.n_dim)(out)
            for i in range(config.n_layers):
                '''
                out = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(config.n_dim, return_sequences=True),
                    backward_layer=tf.keras.layers.GRU(config.n_dim, 
                                                       return_sequences=True,
                                                       go_backwards=True))(out)
                '''
                out = transformer_layer(config.n_dim, config.n_heads)(out)

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
                      loss=focal_loss, # 'binary_crossentropy',
                      metrics=['AUC'])
        model.summary()
        
        if config.pretrain:
            model.load_weights(NAME)
            print('loaded pretrained model')

        """ DATA """
        train_set = make_dataset(config, training=True)
        test_set = make_dataset(config, training=False)

        """ TRAINING """
        callbacks = [
            CSVLogger(NAME.replace('.h5', '.log'),
                      append=True),
            LearningRateScheduler(custom_scheduler(config.n_dim*8, 
                                                   TOTAL_EPOCH/10)),
            SWA(start_epoch=TOTAL_EPOCH//2, swa_freq=2),
            ModelCheckpoint(NAME,
                            monitor='val_auc',
                            mode='max',
                            save_best_only=True),
            TerminateOnNaN()
        ]

        model.fit(train_set,
                  epochs=TOTAL_EPOCH,
                  batch_size=config.batch_size,
                  steps_per_epoch=config.steps_per_epoch,
                  validation_data=test_set,
                  validation_steps=24,
                  callbacks=callbacks)

        # TODO : BN 
        model.save(NAME.replace('.h5', '_SWA.h5'))
