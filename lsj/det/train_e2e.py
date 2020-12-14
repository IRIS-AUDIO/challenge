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
from metrics import *
from models import transformer_layer, encoder


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--pretrain', type=bool, default=False)
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dim', type=int, default=128)
args.add_argument('--n_heads', type=int, default=8)

# DATA
args.add_argument('--background_sounds', type=str,
                  default='/codes/generate_wavs/drone_normed_complex_v2.pickle')
args.add_argument('--voices', type=str,
                  default='/codes/generate_wavs/voice_normed_complex.pickle')
args.add_argument('--labels', type=str,
                  default='/codes/generate_wavs/voice_labels_mfc.npy')
args.add_argument('--noises', type=str,
                  default='/codes/RDChallenge/tf_codes/sounds/noises_specs.pickle')
args.add_argument('--test_background_sounds', type=str,
                  default='/codes/generate_wavs/test_drone_normed_complex.pickle')
args.add_argument('--test_voices', type=str,
                  default='/codes/generate_wavs/test_voice_normed_complex.pickle')
args.add_argument('--test_labels', type=str,
                  default='/codes/generate_wavs/test_voice_labels_mfc.npy')
args.add_argument('--n_mels', type=int, default=128)
args.add_argument('--n_classes', type=int, default=30)

# TRAINING
args.add_argument('--optimizer', type=str, default='sgd',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=0.01)
args.add_argument('--lr_factor', type=float, default=0.7)
args.add_argument('--lr_patience', type=int, default=10)
args.add_argument('--clipvalue', type=float, default=0.01)

args.add_argument('--epochs', type=int, default=400)
args.add_argument('--batch_size', type=int, default=16)
args.add_argument('--n_frame', type=int, default=2048)
args.add_argument('--steps_per_epoch', type=int, default=200)
args.add_argument('--l2', type=float, default=1e-6)

# AUGMENTATION
args.add_argument('--snr', type=float, default=-15)
args.add_argument('--max_voices', type=int, default=6)
args.add_argument('--max_noises', type=int, default=3)


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
    specs = mask(specs, axis=time_axis, max_mask_size=8, n_mask=6) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=8) # freq
    # specs = random_shift(specs, axis=freq_axis, width=8)
    return specs, labels


def load_data(path):
    if path.endswith('.pickle'):
        return pickle.load(open(path, 'rb'))
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('invalid file format')


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
    labels = np.eye(config.n_classes, dtype='float32')[labels] # to one-hot vectors
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
    pipeline = pipeline.map(to_class_labels)
    if training:
        pipeline = pipeline.map(augment)
    pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    pipeline = pipeline.map(minmax_log_on_mel)
    return pipeline.prefetch(AUTOTUNE)


def d_dir(y_true, y_pred, apply_round=True):
    y_true = tf.reshape(y_true, [-1, 3, 10])
    y_pred = tf.reshape(y_pred, [-1, 3, 10])
    if apply_round:
        y_pred = tf.math.round(y_pred)
    return D_direction(tf.reduce_sum(y_true, axis=-2),
                       tf.reduce_sum(y_pred, axis=-2))


def d_cls(y_true, y_pred, apply_round=True):
    y_true = tf.reshape(y_true, [-1, 3, 10])
    y_pred = tf.reshape(y_pred, [-1, 3, 10])
    if apply_round:
        y_pred = tf.math.round(y_pred)
    return D_class(tf.reduce_sum(y_true, axis=-1),
                   tf.reduce_sum(y_pred, axis=-1))


def d_total(y_true, y_pred, apply_round=True):
    return 0.8 * d_dir(y_true, y_pred, apply_round) \
         + 0.2 * d_cls(y_true, y_pred, apply_round)


def mse(y_true, y_pred):
    return d_total(y_true, y_pred, apply_round=False)


def baseline(y_true, y_pred):
    return d_total(y_true, tf.zeros_like(y_true))


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'

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
    out = tf.keras.layers.Dense(config.n_dim)(out)
    out = encoder(config.n_layers,
                  config.n_dim,
                  config.n_heads)(out)

    # out = tf.keras.layers.GlobalAveragePooling1D()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Dense(30, activation=None)(out)
    model = tf.keras.models.Model(inputs=model.input, outputs=out)

    if config.optimizer == 'adam':
        opt = Adam(config.lr, clipvalue=config.clipvalue)
    elif config.optimizer == 'sgd':
        opt = SGD(config.lr, momentum=0.9, clipvalue=config.clipvalue)
    else:
        opt = RMSprop(config.lr, momentum=0.9, clipvalue=config.clipvalue)

    if config.l2 > 0:
        model = apply_kernel_regularizer(
            model, tf.keras.regularizers.l2(config.l2))
    model.compile(optimizer=opt, 
                  loss=mse, # 'binary_crossentropy',
                  metrics=[d_total, baseline])
    model.summary()
    

    """ DATA """
    train_set = make_dataset(config, training=True)
    test_set = make_dataset(config, training=False)
    print(train_set)
    for x, y in train_set.take(1):
        print(tf.shape(x), tf.shape(y))

    """ TRAINING """
    from train_frame import custom_scheduler
    callbacks = [
        CSVLogger(NAME.replace('.h5', '.log'),
                  append=True),
        # LearningRateScheduler(custom_scheduler(config.n_dim*8, config.epochs/10)),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=config.lr_factor,
                          patience=config.lr_patience),
        SWA(start_epoch=config.epochs//2, swa_freq=2),
        ModelCheckpoint(NAME,
                        monitor='val_d_total',
                        mode='min',
                        save_best_only=True),
        TerminateOnNaN()
    ]

    model.fit(train_set,
              epochs=config.epochs,
              batch_size=config.batch_size,
              steps_per_epoch=config.steps_per_epoch,
              validation_data=test_set,
              validation_steps=16,
              callbacks=callbacks)

    # TODO : BN 
    model.save(NAME.replace('.h5', '_SWA.h5'))
