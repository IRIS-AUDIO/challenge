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
from metrics import *
from pipeline import *
from swa import SWA
from transforms import *
from utils import *
from EfficientDet.model import myefficientdet


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB6')
args.add_argument('--pretrain', type=bool, default=False)
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dim', type=int, default=64) # 256)
args.add_argument('--n_heads', type=int, default=8)
args.add_argument('--m_type', type=int, default=1)

abspath = '/root/otherperson/daniel'
# DATA
args.add_argument('--background_sounds', type=str,
                  default=abspath+'/generate_wavs/drone_normed_complex_v3.pickle')
args.add_argument('--voices', type=str,
                  default=abspath+'/generate_wavs/voice_normed_complex_v3.pickle')
args.add_argument('--labels', type=str,
                  default=abspath+'/generate_wavs/voice_labels_mfc_v3.npy')
args.add_argument('--noises', type=str,
                  default=abspath+'/RDChallenge/tf_codes/sounds/noises_specs.pickle')
args.add_argument('--test_background_sounds', type=str,
                  default=abspath+'/generate_wavs/test_drone_normed_complex_v2.pickle')
args.add_argument('--test_voices', type=str,
                  default=abspath+'/generate_wavs/test_voice_normed_complex.pickle')
args.add_argument('--test_labels', type=str,
                  default=abspath+'/generate_wavs/test_voice_labels_mfc.npy')
args.add_argument('--n_mels', type=int, default=128)

# TRAINING
args.add_argument('--optimizer', type=str, default='adam',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=1e-4)
args.add_argument('--end_lr', type=float, default=1e-4)
args.add_argument('--lr_power', type=float, default=0.5)
args.add_argument('--lr_div', type=float, default=3) # 2)
args.add_argument('--clipvalue', type=float, default=0.01)

args.add_argument('--epochs', type=int, default=500)
args.add_argument('--batch_size', type=int, default=12)
args.add_argument('--n_frame', type=int, default=2048)
args.add_argument('--steps_per_epoch', type=int, default=100)
args.add_argument('--l1', type=float, default=0)
args.add_argument('--l2', type=float, default=1e-6)
args.add_argument('--loss_l2', type=float, default=1.)
args.add_argument('--multiplier', type=float, default=10)

# AUGMENTATION
args.add_argument('--snr', type=float, default=-15)
args.add_argument('--max_voices', type=int, default=10)
args.add_argument('--max_noises', type=int, default=6)


def minmax_log_on_mel(mel, labels=None):
    axis = tuple(range(1, len(mel.shape)))

    # MIN-MAX
    mel_max = tf.math.reduce_max(mel, axis=axis, keepdims=True)
    mel_min = tf.math.reduce_min(mel, axis=axis, keepdims=True)
    mel = safe_div(mel-mel_min, mel_max-mel_min)

    # LOG
    mel = tf.math.log(mel + EPSILON)

    if labels is not None:
        return mel, labels
    return mel


def random_reverse_chan(specs, labels):
    # Assume MelSpectrogram
    # specs: [..., freq, time, chan]
    # labels: [..., time', 30]
    rev_specs = tf.reverse(specs, [-1])
    rev_labels = tf.stack(
        tf.split(labels, 3, axis=-1), axis=-2) # [..., time', 3, 10]
    rev_labels = tf.reverse(rev_labels, [-1])
    rev_labels = tf.concat(
        tf.unstack(rev_labels, axis=-2), axis=-1)

    coin = tf.cast(tf.random.uniform([]) > 0.5, tf.float32)

    specs = coin * specs + (1-coin) * rev_specs
    labels = coin * labels + (1-coin) * rev_labels
                          
    return specs, labels


def augment(specs, labels, time_axis=-2, freq_axis=-3):
    specs = mask(specs, axis=time_axis, max_mask_size=16, n_mask=6) 
    specs = mask(specs, axis=freq_axis, max_mask_size=12)
    specs, labels = random_reverse_chan(specs, labels)
    return specs, labels


def preprocess_labels(multiplier):
    def _preprocess(x, y):
        # process y: [None, time, classes] -> [None, time', classes]
        for i in range(3): # 5):
            # sum_pool1d
            y = tf.nn.avg_pool1d(y, 2, strides=2, padding='SAME') * 2
        y *= multiplier
        return x, y
    return _preprocess


def to_density_labels(x, y):
    """
    :param y: [..., n_voices, n_frames, n_classes]
    :return: [..., n_frames, n_classes]
    """
    y = safe_div(y, tf.reduce_sum(y, axis=(-2, -1), keepdims=True))
    y = tf.reduce_sum(y, axis=-3)
    return x, y


def magphase_to_mag(x, y=None):
    x = x[..., :tf.shape(x)[-1] // 2] # remove phase
    if y is None:
        return x
    return x, y

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
    labels = np.eye(30, dtype='float32')[labels] # to one-hot vectors
    noises = load_data(config.noises)

    # Make pipeline and process the pipeline
    pipeline = make_pipeline(backgrounds, 
                             voices, labels,
                             noises,
                             n_frame=config.n_frame,
                             max_voices=config.max_voices,
                             max_noises=config.max_noises,
                             n_classes=30,
                             snr=config.snr,
                             # voice_map_fn=random_speedup(stddev=0.05) if training else None, # TEST (D_B4_B3)
                             min_ratio=1)
    pipeline = pipeline.map(to_density_labels)
    if training: 
        pipeline = pipeline.map(augment)
    pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    # pipeline = pipeline.map(magphase_to_mag) 
    pipeline = pipeline.map(minmax_log_on_mel)
    pipeline = pipeline.map(preprocess_labels(config.multiplier))
    return pipeline.prefetch(AUTOTUNE)


def d_total(multiplier=10):
    def d_total(y_true, y_pred, apply_round=True):
        y_true /= multiplier
        y_pred /= multiplier

        # [None, time, 30] -> [None, time, 3, 10]
        y_true = tf.stack(tf.split(y_true, 3, axis=-1), axis=-2)
        y_pred = tf.stack(tf.split(y_pred, 3, axis=-1), axis=-2)

        # d_dir
        d_true = tf.reduce_sum(y_true, axis=(-3, -2))
        d_pred = tf.reduce_sum(y_pred, axis=(-3, -2))
        if apply_round:
            d_true = tf.math.round(d_true)
            d_pred = tf.math.round(d_pred)
        d_dir = D_direction(d_true, d_pred)

        # c_cls
        c_true = tf.reduce_sum(y_true, axis=(-3, -1))
        c_pred = tf.reduce_sum(y_pred, axis=(-3, -1))
        if apply_round:
            c_true = tf.math.round(c_true)
            c_pred = tf.math.round(c_pred)
        d_cls = D_class(c_true, c_pred)

        return 0.8 * d_dir + 0.2 * d_cls
    return d_total


def custom_loss(alpha=0.8, l2=1, multiplier=10):
    total_loss = d_total(multiplier)

    def _custom(y_true, y_pred):
        # y_true, y_pred = [None, time, 30]
        # [None, time, 30] -> [None, time, 3, 10]
        t_true = tf.stack(tf.split(y_true, 3, axis=-1), axis=-2)
        t_pred = tf.stack(tf.split(y_pred, 3, axis=-1), axis=-2)

        # [None, time, 10]
        d_y_true = tf.reduce_sum(t_true, axis=-2)
        d_y_pred = tf.reduce_sum(t_pred, axis=-2)

        # [None, time, 3]
        c_y_true = tf.reduce_sum(t_true, axis=-1)
        c_y_pred = tf.reduce_sum(t_pred, axis=-1)

        loss = alpha * tf.keras.losses.MAE(tf.reduce_sum(d_y_true, axis=1),
                                           tf.reduce_sum(d_y_pred, axis=1)) \
             + (1-alpha) * tf.keras.losses.MAE(tf.reduce_sum(c_y_true, axis=1),
                                               tf.reduce_sum(c_y_pred, axis=1))

        # TODO: OT loss
        # TV: total variation loss
        # normed - degrees [None, time, 10]
        n_d_true = safe_div(
            d_y_true, tf.reduce_sum(d_y_true, axis=1, keepdims=True))
        n_d_pred = safe_div(
            d_y_pred, tf.reduce_sum(d_y_pred, axis=1, keepdims=True))

        # normed - classes [None, time, 3]
        n_c_true = safe_div(
            c_y_true, tf.reduce_sum(c_y_true, axis=1, keepdims=True))
        n_c_pred = safe_div(
            c_y_pred, tf.reduce_sum(c_y_pred, axis=1, keepdims=True))

        tv = alpha * tf.reduce_mean(
                tf.reduce_sum(tf.math.abs(n_d_true - n_d_pred), axis=1) 
                * tf.reduce_sum(d_y_true, axis=1), # [None, 10]
                axis=1)
        tv += (1-alpha) * tf.reduce_mean(
                tf.reduce_sum(tf.math.abs(n_c_true - n_c_pred), axis=1) 
                * tf.reduce_sum(c_y_true, axis=1), # [None, 3]
                axis=1)
        loss += l2 * tv

        return loss
    return _custom


def cos_sim(y_true, y_pred):
    mask = tf.cast(
        tf.reduce_sum(y_true, axis=-2) > 0., tf.float32) # [None, 30]
    mask = safe_div(mask, tf.reduce_sum(mask, axis=-1, keepdims=True))
    return tf.reduce_sum(
        tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-2) * mask, 
        axis=-1)


def polydecay_with_warmup(base_lr, end_lr, 
                          decay_steps, warmup_steps,
                          power=0.5):
    # lr: maximum lr
    poly_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        base_lr, decay_steps, end_lr, power)

    def _scheduler(step):
        arg1 = poly_scheduler(tf.maximum(0, step-warmup_steps))
        arg2 = (step+1) * base_lr / warmup_steps
        return tf.math.minimum(arg1, arg2)
    return _scheduler


def custom_scheduler(d_model, warmup_steps=4000, lr_div=2):
    # https://www.tensorflow.org/tutorials/text/transformer#optimizer
    d_model = tf.cast(d_model, tf.float32)

    def _scheduler(step):
        step = tf.cast(step+1, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (warmup_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2) / lr_div
    return _scheduler


def random_speedup(stddev=0.1):
    def _random_speedup(voice, label=None):
        voice = phase_vocoder(voice, rate=tf.random.normal([], mean=1., stddev=stddev))
        if label is None:
            return voice
        return voice, label
    return _random_speedup


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'

    """ MODEL """
    model = myefficientdet(phi=int(config.model[-1]),
                           input_shape=(config.n_mels, None, 2),
                           n_dims=config.n_dim,
                           m_type=config.m_type,
                           backend=tf.keras.backend,
                           layers=tf.keras.layers,
                           models=tf.keras.models,
                           utils=tf.keras.utils,
                           )

    lr = config.lr
    if config.optimizer == 'adam':
        opt = Adam(lr, clipvalue=config.clipvalue)
    elif config.optimizer == 'sgd':
        opt = SGD(lr, momentum=0.9, clipvalue=config.clipvalue)
    else:
        opt = RMSprop(lr, momentum=0.9, clipvalue=config.clipvalue)

    if config.l2 > 0:
        model = apply_kernel_regularizer(
            model, tf.keras.regularizers.l1_l2(config.l1, config.l2))
    model.compile(optimizer=opt, 
                  loss=custom_loss(alpha=0.8, l2=config.loss_l2),
                  metrics=[d_total(config.multiplier), cos_sim])
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
        SWA(start_epoch=TOTAL_EPOCH//2, swa_freq=2),
        ModelCheckpoint(NAME,
                        monitor='val_d_total', 
                        save_best_only=True,
                        verbose=1),
        TerminateOnNaN(),
        TensorBoard(log_dir='tensorboard_log/'+config.name)
    ]

    if not config.pretrain:
        callbacks.append(
            LearningRateScheduler(
                custom_scheduler(4096, TOTAL_EPOCH/12, config.lr_div)))
    else:
        callbacks.append(
            ReduceLROnPlateau(monitor='d_total', factor=0.9, patience=5))

    model.fit(train_set,
              epochs=TOTAL_EPOCH,
              batch_size=BATCH_SIZE,
              steps_per_epoch=config.steps_per_epoch,
              validation_data=test_set,
              validation_steps=16,
              callbacks=callbacks)

    model.save(NAME.replace('.h5', '_SWA.h5'))
