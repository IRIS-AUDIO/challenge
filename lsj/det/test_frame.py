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


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dim', type=int, default=256)
args.add_argument('--n_heads', type=int, default=8)

# DATA
args.add_argument('--background_sounds', type=str,
                  default='/codes/generate_wavs/drone_normed_complex.pickle')
args.add_argument('--voices', type=str,
                  default='/codes/generate_wavs/voice_normed_complex.pickle')
args.add_argument('--labels', type=str,
                  default='/codes/generate_wavs/voice_labels_mfc.npy')
args.add_argument('--noises', type=str,
                  default='/codes/RDChallenge/tf_codes/sounds/noises_specs_2.pickle')
args.add_argument('--n_mels', type=int, default=100)

# TRAINING
args.add_argument('--optimizer', type=str, default='adam',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--lr_factor', type=float, default=0.7)
args.add_argument('--lr_patience', type=int, default=10)

args.add_argument('--epochs', type=int, default=500)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--n_frame', type=int, default=2000)
args.add_argument('--steps_per_epoch', type=int, default=200)
args.add_argument('--l2', type=float, default=1e-6)

# AUGMENTATION
args.add_argument('--alpha', type=float, default=0.75)
args.add_argument('--snr', type=float, default=-10)
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
        # x = tf.keras.layers.Input(shape=(257, None, 4))
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
                      loss='binary_crossentropy',
                      metrics=['AUC'])
        model.summary()
        model.load_weights(NAME)
        print('loaded pretrained model')

        """ DATA """
        # wavs = glob.glob('/codes/2020_track3/t3_audio/*.wav')
        wavs = glob.glob('/media/data1/datasets/ai_challenge/2020_track3/t3_audio/*.wav')
        wavs.sort()
        to_mel = magphase_to_mel(config.n_mels)

        for wav in wavs:
            sample = load_wav(wav)[None, :] # [1, freq, time, chan2]
            sample = complex_to_magphase(sample)
            sample = to_mel(sample)
            sample = minmax_log_on_mel(sample)

            # PREDICT
            output = model.predict(sample)[0] # [time', 30]
            plt.imshow(output)
            plt.savefig(os.path.split(wav)[-1].replace('.wav', '.png'))
