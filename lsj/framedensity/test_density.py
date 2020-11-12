import argparse
import glob
import numpy as np
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

import efficientnet.model as model
from swa import SWA
from pipeline import *
from transforms import *
from utils import *
from data_utils import *
from metrics import *
from models import transformer_layer, encoder

np.set_printoptions(precision=4)

args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB4')
args.add_argument('--mode', type=str, default='GRU',
                                 choices=['GRU', 'transformer'])
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dim', type=int, default=256)
args.add_argument('--n_heads', type=int, default=8)

args.add_argument('--n_mels', type=int, default=128)
args.add_argument('--n_classes', type=int, default=30)

# DATA
args.add_argument('--background_sounds', type=str,
                  default='../generate_wavs/test_drone_normed_complex.pickle')
args.add_argument('--voices', type=str,
                  default='../generate_wavs/test_voice_normed_complex.pickle')
args.add_argument('--labels', type=str,
                  default='../generate_wavs/test_voice_labels_mfc.npy')
args.add_argument('--noises', type=str,
                  default='../RDChallenge/tf_codes/sounds/noises_specs.pickle')
# TRAINING

args.add_argument('--epochs', type=int, default=500)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--n_frame', type=int, default=2048)
args.add_argument('--steps_per_epoch', type=int, default=200)
args.add_argument('--l2', type=float, default=1e-6)

# AUGMENTATION
args.add_argument('--alpha', type=float, default=0.75)
args.add_argument('--snr', type=float, default=-10)
args.add_argument('--max_voices', type=int, default=6)
args.add_argument('--max_noises', type=int, default=3)

args.add_argument('--reverse', type=bool, default=True)
args.add_argument('--multiplier', type=float, default=10.)


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


def load_data(path):
    if path.endswith('.pickle'):
        return pickle.load(open(path, 'rb'))
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('invalid file format')


def d_dir(y_true, y_pred):
    # [None, time, 30] -> [None, time, 3, 10]
    y_true = tf.stack(tf.split(y_true, 3, axis=-1), axis=-2)
    y_pred = tf.stack(tf.split(y_pred, 3, axis=-1), axis=-2)

    y_true = tf.math.round(tf.reduce_sum(y_true, axis=(-3, -2)))
    y_pred = tf.math.round(tf.reduce_sum(y_pred, axis=(-3, -2)))
    return D_direction(y_true, y_pred)


def d_cls(y_true, y_pred):
    # [None, time, 30] -> [None, time, 3, 10]
    y_true = tf.stack(tf.split(y_true, 3, axis=-1), axis=-2)
    y_pred = tf.stack(tf.split(y_pred, 3, axis=-1), axis=-2)

    y_true = tf.math.round(tf.reduce_sum(y_true, axis=(-3, -1)))
    y_pred = tf.math.round(tf.reduce_sum(y_pred, axis=(-3, -1)))
    return D_class(y_true, y_pred)


def d_total(y_true, y_pred):
    return 0.8 * d_dir(y_true, y_pred) + 0.2 * d_cls(y_true, y_pred)


def to_density_labels(x, y):
    """
    :param y: [n_voices, n_frames, n_classes]
    :return: [n_frames, n_classes]
    """
    y = y / (tf.reduce_sum(y, axis=(1, 2), keepdims=True) + EPSILON)
    y = tf.reduce_sum(y, axis=0)
    return x, y


def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=16, n_mask=6) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=32) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    return specs, labels


def preprocess_labels(x, y):
    # preprocess y
    for i in range(5):
        # sum_pool1d
        y = tf.nn.avg_pool1d(y, 2, strides=2, padding='SAME') * 2
    # y = mu_law(y)
    return x, y


def make_dataset(config, training=False):
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
    pipeline = pipeline.map(to_density_labels)
    # if training:
    #     pipeline = pipeline.map(augment)
    pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    pipeline = pipeline.map(minmax_log_on_mel)
    pipeline = pipeline.map(preprocess_labels)
    return pipeline.prefetch(AUTOTUNE)


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
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

    if config.n_layers > 0:
        if config.mode == 'GRU':
            out = tf.keras.layers.Dense(config.n_dim)(out)
            for i in range(config.n_layers):
                # out = transformer_layer(config.n_dim, config.n_heads)(out)
                out = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(config.n_dim, return_sequences=True),
                    backward_layer=tf.keras.layers.GRU(config.n_dim, 
                                                       return_sequences=True,
                                                       go_backwards=True))(out)
        elif config.mode == 'transformer':
            out = tf.keras.layers.Dense(config.n_dim)(out)
            out = encoder(config.n_layers,
                          config.n_dim,
                          config.n_heads)(out)

            out = tf.keras.layers.Flatten()(out)
            out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Dense(config.n_classes, activation='relu')(out)
    model = tf.keras.models.Model(inputs=model.input, outputs=out)

    model.load_weights(NAME)
    print('loaded pretrained model')

    """ DATA """
    # wavs = glob.glob('/codes/2020_track3/t3_audio/*.wav')
    wavs = glob.glob('/root/datasets/ai_challenge/2020_track3/t3_audio/*.wav')
    wavs.sort()
    to_mel = magphase_to_mel(config.n_mels)
    
    '''

    train_set = make_dataset(config, training=True)
    for x, y in train_set.take(1):
        p = model.predict(x)
        p /= config.multiplier
        print(d_total(y, p))

        for i in range(5):
            if config.mode == 'GRU':
                # PREDICTION
                output = tf.reduce_sum(p[i], axis=0) # [30]
                output = tf.reshape(output, (3, 10))
                print(tf.reduce_sum(output, axis=0).numpy().round(),
                      tf.reduce_sum(output, axis=1).numpy().round())

                # TRUE
                output = np.sum(y[i], axis=0).reshape(3, 10)
                print(np.sum(output, axis=0).round(),
                      np.sum(output, axis=1).round())

                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(y[i])
                axs[1].imshow(p[i])
                axs[2].imshow(x[i][..., 0].numpy().T)
                plt.show()
            elif config.mode == 'transformer':
                print(p[i])
                print(np.sum(y[i], axis=0))
    '''

    gt_angle = [[0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                [2, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                [1, 2, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0]]
    gt_class = [[1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 3],
                [0, 1, 2],
                [2, 1, 0],
                [2, 2, 1],
                [0, 1, 2],
                [2, 1, 1]]
    
    print(wavs)
    wavs = list(map(load_wav, wavs)) 
    target = max([tuple(wav.shape) for wav in wavs]) 
    wavs = list(map(lambda x: tf.pad(x, [[0, 0], [0, target[1]-x.shape[1]], [0, 0]]), 
                    wavs)) 
    wavs = tf.convert_to_tensor(wavs) 
    wavs = complex_to_magphase(wavs) 
    wavs = magphase_to_mel(config.n_mels)(wavs) 
    wavs = minmax_log_on_mel(wavs) 
    wavs = model.predict(wavs) 
    wavs = wavs / config.multiplier
    wavs = tf.reshape(wavs, [*wavs.shape[:2], 3, 10])

    angles = tf.round(tf.reduce_sum(wavs, axis=(1, 2)))
    classes = tf.round(tf.reduce_sum(wavs, axis=(1, 3)))

    d_dir = D_direction(tf.cast(gt_angle, tf.float32), 
                        tf.cast(angles, tf.float32))
    d_cls = D_class(tf.cast(gt_class, tf.float32),
                    tf.cast(classes, tf.float32))

    d_total = (d_dir * 0.8 + d_cls * 0.2).numpy()
    print('total')
    print(d_total, d_total.mean())

    for i in range(len(gt_angle)):
        # plt.imshow(wav); plt.show()
        print(angles[i].numpy(), classes[i].numpy())
        print(gt_angle[i], gt_class[i])
        print()
