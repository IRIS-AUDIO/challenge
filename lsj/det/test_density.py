import argparse
import glob
import json
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
from data_utils import *
from metrics import *
from models import transformer_layer, encoder

np.set_printoptions(precision=4)

args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB4')
args.add_argument('--mode', type=str, default='Dense',
                                 choices=['Dense', 'GRU', 'transformer'])
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dim', type=int, default=256)
args.add_argument('--n_heads', type=int, default=8)

args.add_argument('--n_mels', type=int, default=128)
args.add_argument('--n_classes', type=int, default=30)

# DATA
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--data', type=str, default='challenge',
                  choices=['challenge', 'SKKU', 'GIST', 'KICT', '2019'])
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


def load_json(path):
    gt = json.load(open(path, 'rb'))['track3_results']
    gt.sort(key=lambda x: x['id'])
    angles = np.stack([x['angle'] for x in gt])
    classes = np.stack([x['class'] for x in gt])
    return angles, classes


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    """ DATA """
    DATA = config.data
    if DATA == 'challenge':
        wavs = glob.glob('/media/data1/datasets/ai_challenge/2020_track3/t3_audio/*.wav')
        gt_angle, gt_class = load_json('/media/data1/datasets/ai_challenge/2020_track3/t3_res_sample.json')
    else: # if DATA in ['SKKU', 'GIST', 'KICT']:
        wavs = glob.glob(f'/media/data1/datasets/ai_challenge/2020_validation/{DATA}/wavs/*.wav')
        gt_angle, gt_class = load_json(f'/media/data1/datasets/ai_challenge/2020_validation/{DATA}/labels.json')
    wavs.sort()
    
    print(wavs[:20])
    wavs = list(map(load_wav, wavs))
    target = max([tuple(wav.shape) for wav in wavs])
    wavs = list(map(lambda x: tf.pad(x, [[0, 0], [0, target[1]-x.shape[1]], [0, 0]]),
                    wavs))
    wavs = tf.convert_to_tensor(wavs)
    wavs = complex_to_magphase(wavs)
    wavs = magphase_to_mel(config.n_mels)(wavs)
    wavs = minmax_log_on_mel(wavs)
    if DATA == 'challenge':
        wavs = tf.concat([wavs, tf.reverse(wavs, axis=[-1])], axis=0)
        gt_angle = tf.concat([gt_angle, tf.reverse(gt_angle, axis=[-1])], axis=0)
        gt_angle = gt_angle.numpy()
        gt_class = tf.concat([gt_class, gt_class], axis=0)
        gt_class = gt_class.numpy()

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
        else:
            for i in range(config.n_layers):
                # out = tf.keras.layers.Dropout(0.1)(out)
                out = tf.keras.layers.Dense(config.n_dim)(out)
                out = tf.keras.layers.Activation('sigmoid')(out) * out

    out = tf.keras.layers.Dense(config.n_classes, activation='relu')(out)
    model = tf.keras.models.Model(inputs=model.input, outputs=out)

    specs = None
    for name in config.name.split(','):
        NAME = name if name.endswith('.h5') else name + '.h5'
        model.load_weights(NAME)

        if specs is None:
            specs = model.predict(wavs, verbose=True) 
        else:
            specs += model.predict(wavs, verbose=True)
    specs /= len(config.name.split(','))

    specs = specs / config.multiplier
    specs = tf.reshape(specs, [*specs.shape[:2], 3, 10])

    angles = tf.cast(tf.round(tf.reduce_sum(specs, axis=(1, 2))), tf.int32)
    classes = tf.cast(tf.round(tf.reduce_sum(specs, axis=(1, 3))), tf.int32)

    d_dir = D_direction(tf.cast(gt_angle, tf.float32), 
                        tf.cast(angles, tf.float32))
    d_cls = D_class(tf.cast(gt_class, tf.float32),
                    tf.cast(classes, tf.float32))

    d_total = (d_dir * 0.8 + d_cls * 0.2).numpy()
    print('total')
    print(d_total[:25], d_total.mean())

    for i in range(min(25, len(gt_angle))):
        # plt.imshow(wav); plt.show()
        print(angles[i].numpy(), classes[i].numpy())
        print(gt_angle[i], gt_class[i])
        print()
