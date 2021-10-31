import argparse
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

from metrics import *
from pipeline import *
from swa import SWA, NO_SWA_ERROR
from transforms import *
from utils import *


class ARGS:
    def __init__(self) -> None:
        self.args = argparse.ArgumentParser()
        self.args.add_argument('--name', type=str, default='')
        self.args.add_argument('--gpus', type=str, default='-1')
        self.args.add_argument('--model', type=int, default=0)
        self.args.add_argument('--v', type=int, default=2)
        self.args.add_argument('--pretrain', type=bool, default=False)
        self.args.add_argument('--n_layers', type=int, default=0)
        self.args.add_argument('--n_dim', type=int, default=256)
        self.args.add_argument('--n_chan', type=int, default=2)
        self.args.add_argument('--n_classes', type=int, default=3)

        # DATA
        self.args.add_argument('--datapath', type=str, default='/root/datasets/Interspeech2020/generate_wavs/codes')
        self.args.add_argument('--background_sounds', type=str, default='drone_normed_complex_v3.pickle')
        self.args.add_argument('--voices', type=str, default='voice_normed_complex_v3.pickle')
        self.args.add_argument('--labels', type=str, default='voice_labels_mfc_v3.npy')
        self.args.add_argument('--noises', type=str, default='noises_specs_v2.pickle')
        self.args.add_argument('--test_background_sounds', type=str,
                        default='test_drone_normed_complex_v2.pickle')
        self.args.add_argument('--test_voices', type=str, default='test_voice_normed_complex.pickle')
        self.args.add_argument('--test_labels', type=str, default='test_voice_labels_mfc.npy')
        self.args.add_argument('--n_mels', type=int, default=80)

        # TRAINING
        self.args.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop', 'adabelief'])
        self.args.add_argument('--lr', type=float, default=1e-3)
        self.args.add_argument('--end_lr', type=float, default=1e-4)
        self.args.add_argument('--lr_power', type=float, default=0.5)
        self.args.add_argument('--lr_div', type=float, default=2)
        self.args.add_argument('--clipvalue', type=float, default=0.01)

        self.args.add_argument('--epochs', type=int, default=400)
        self.args.add_argument('--batch_size', type=int, default=12)
        self.args.add_argument('--n_frame', type=int, default=2048)
        self.args.add_argument('--steps_per_epoch', type=int, default=100)
        self.args.add_argument('--l1', type=float, default=0)
        self.args.add_argument('--l2', type=float, default=1e-6)
        self.args.add_argument('--loss_alpha', type=float, default=0.8)
        self.args.add_argument('--loss_l2', type=float, default=1.)
        self.args.add_argument('--multiplier', type=float, default=10)
        self.args.add_argument('--loss', type=str, default='FOCAL')

        # AUGMENTATION
        self.args.add_argument('--snr', type=float, default=-15)
        self.args.add_argument('--max_voices', type=int, default=10)
        self.args.add_argument('--max_noises', type=int, default=6)

    def get(self):
        return self.args.parse_args()


def minmax_log_on_mel(mel, labels=None):
    # batch-wise pre-processing
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


def augment(specs, labels, time_axis=-2, freq_axis=-3):
    specs = mask(specs, axis=time_axis, max_mask_size=24, n_mask=6)
    specs = mask(specs, axis=freq_axis, max_mask_size=16)
    return specs, labels


def to_frame_labels(x, y):
    """
    :param y: [..., n_voices, n_frames, n_classes]
    :return: [..., n_frames, n_classes]
    """
    y = tf.reduce_sum(y, axis=-3)
    return x, y


def mono_chan(x, y):
    return x[..., :1], y


def stereo_mono(x, y=None):
    if y is None:
        return tf.concat([x[..., :2], x[..., :1] + x[..., 1:2], x[..., 2:4], x[..., 2:3] + x[..., 3:4]], -1)
    return tf.concat([x[..., :2], x[..., :1] + x[..., 1:2], x[..., 2:4], x[..., 2:3] + x[..., 3:4]], -1), y


def label_downsample(x, y):
    y = tf.keras.layers.AveragePooling1D(32, 32)(y)
    y = tf.cast(y >= 0.5, y.dtype)
    return x, y


def make_dataset(config, training=True, n_classes=3):
    # Load required datasets
    if not os.path.exists(config.datapath):
        config.datapath = ''
    if training:
        backgrounds = load_data(os.path.join(config.datapath, config.background_sounds))
        voices = load_data(os.path.join(config.datapath, config.voices))
        labels = load_data(os.path.join(config.datapath, config.labels))
    else:
        backgrounds = load_data(os.path.join(config.datapath, config.test_background_sounds))
        voices = load_data(os.path.join(config.datapath, config.test_voices))
        labels = load_data(os.path.join(config.datapath, config.test_labels))
    if labels.max() - 1 != config.n_classes:
        labels //= 10
    labels = np.eye(n_classes, dtype='float32')[labels] # to one-hot vectors
    noises = load_data(os.path.join(config.datapath, config.noises))

    # Make pipeline and process the pipeline
    pipeline = make_pipeline(backgrounds, 
                             voices, labels, noises,
                             n_frame=config.n_frame,
                             max_voices=config.max_voices,
                             max_noises=config.max_noises,
                             n_classes=n_classes,
                             snr=config.snr,
                             min_ratio=1)
    pipeline = pipeline.map(to_frame_labels)
    if training: 
        pipeline = pipeline.map(augment)
    if config.n_chan == 1:
        pipeline = pipeline.map(mono_chan)
    elif config.n_chan == 3:
        pipeline = pipeline.map(stereo_mono)
    pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    pipeline = pipeline.map(minmax_log_on_mel)
    if config.v == 3:
        pipeline = pipeline.map(label_downsample)
    return pipeline.prefetch(AUTOTUNE)


def custom_loss(alpha=0.8, l2=1):
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
        tf.reduce_sum(y_true, axis=-2) > 0., tf.float32) # [None, 3]
    mask = safe_div(mask, tf.reduce_sum(mask, axis=-1, keepdims=True))
    return tf.reduce_sum(
        tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-2) * mask, 
        axis=-1)


def custom_scheduler(d_model, warmup_steps=4000, lr_div=2):
    # https://www.tensorflow.org/tutorials/text/transformer#optimizer
    d_model = tf.cast(d_model, tf.float32)

    def _scheduler(step):
        step = tf.cast(step+1, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (warmup_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2) / lr_div
    return _scheduler


def get_model(config):
    input_tensor = tf.keras.layers.Input(
        shape=(config.n_mels, config.n_frame, config.n_chan))
    backbone = getattr(tf.keras.applications.efficientnet, f'EfficientNetB{config.model}')(
        include_top=False, weights=None, input_tensor=input_tensor)

    out = tf.transpose(backbone.output, perm=[0, 2, 1, 3])
    out = tf.keras.layers.Reshape([-1, out.shape[-1]*out.shape[-2]])(out)

    for i in range(config.n_layers):
        out = tf.keras.layers.Dense(config.n_dim)(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation('sigmoid')(out) * out

    # v1 -------------------------
    if config.v == 1:
        out = tf.keras.layers.Conv1DTranspose(128, 2, 2)(out)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Conv1DTranspose(64, 2, 2)(out)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Conv1DTranspose(32, 2, 2)(out)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Conv1DTranspose(16, 2, 2)(out)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Conv1DTranspose(3, 2, 2)(out)
        out = tf.keras.layers.Activation('relu')(out)

    # v2 -------------------------
    elif config.v == 2:
        out = tf.keras.layers.Conv1DTranspose(128, 2, 2)(out)
        out = tf.keras.layers.Conv1DTranspose(64, 2, 2)(out)
        out = tf.keras.layers.Conv1DTranspose(32, 2, 2)(out)
        out = tf.keras.layers.Conv1DTranspose(16, 2, 2)(out)
        out = tf.keras.layers.Conv1DTranspose(3, 2, 2)(out)
    elif config.v == 3:
        out = out
    elif config.v == 4:
        out = tf.keras.layers.Conv1D(config.n_frame, 1, use_bias=False, data_format='channels_first')(out)
    else:
        raise ValueError('wrong version')
    
    out = tf.keras.layers.Dense(config.n_classes)(out)
    # out= tf.keras.layers.Activation('relu')(out)
    # out *= tf.cast(out < 1., out.dtype)
    out = tf.keras.layers.Activation('sigmoid')(out)
    return tf.keras.models.Model(inputs=input_tensor, outputs=out)


def main():
    config = ARGS().get()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    config.loss = config.loss.upper()
    print(config)

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = (config.name + '_') if config.name != '' else ''
    NAME = NAME + '_'.join([f'B{config.model}', f'v{config.v}', f'lr{config.lr}', 
                            f'batch{config.batch_size}', f'opt_{config.optimizer}', 
                            f'mel{config.n_mels}', f'chan{config.n_chan}', f'{config.loss.upper()}'])
    NAME = NAME if NAME.endswith('.h5') else NAME + '.h5'
    """ MODEL """
    model = get_model(config)

    lr = config.lr
    if config.optimizer == 'adam':
        opt = Adam(lr, clipvalue=config.clipvalue)
    elif config.optimizer == 'sgd':
        opt = SGD(lr, momentum=0.9, clipvalue=config.clipvalue)
    elif config.optimizer == 'rmsprop':
        opt = RMSprop(lr, momentum=0.9, clipvalue=config.clipvalue)
    else:
        opt = AdaBelief(lr, clipvalue=config.clipvalue)
    if config.l2 > 0:
        model = apply_kernel_regularizer(
            model, tf.keras.regularizers.l1_l2(config.l1, config.l2))

    if config.loss.upper() == 'BCE':
        loss = tf.keras.losses.BinaryCrossentropy()
    elif config.loss.upper() == 'FOCAL':
        loss = sigmoid_focal_crossentropy

    model.compile(optimizer=opt, 
                #   loss=custom_loss(alpha=config.loss_alpha, l2=config.loss_l2),
                  loss=loss,
                  metrics=[tf.keras.metrics.CosineSimilarity(name='tf_cos_sim', axis=-2),
                           cos_sim,
                           tfa.metrics.F1Score(num_classes=3, threshold=0.5, average='micro'),
                           get_custom_er_new])
    model.summary()
    print(NAME)

    if config.pretrain:
        model.load_weights(NAME)
        print('loaded pretrained model')

    """ DATA """
    train_set = make_dataset(config, training=True)
    test_set = make_dataset(config, training=False)

    """ TRAINING """
    callbacks = [
        CSVLogger(NAME.replace('.h5', '.csv'), append=True),
        SWA(start_epoch=TOTAL_EPOCH//4, swa_freq=2),
        ModelCheckpoint(NAME, monitor='val_loss', save_best_only=True, verbose=1),
        TerminateOnNaN(),
        TensorBoard(log_dir=os.path.join('tensorboard_log', NAME.split('.h5')[0])),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    ]

    if not config.pretrain:
        callbacks.append(
            LearningRateScheduler(
                custom_scheduler(4096, TOTAL_EPOCH/12, config.lr_div)))
    else:
        callbacks.append(
            ReduceLROnPlateau(monitor='val_loss', factor=1 / 2**0.5, patience=5, verbose=1, mode='min'))

    try:
        from time import time
        st = time()
        model.fit(train_set,
                epochs=TOTAL_EPOCH,
                batch_size=BATCH_SIZE,
                steps_per_epoch=config.steps_per_epoch,
                validation_data=test_set,
                validation_steps=16,
                callbacks=callbacks)
        print('best model:', NAME.replace('.h5', '_SWA.h5'))
        print(time() - st, 'seconds')
        model.save(NAME.replace('.h5', '_SWA.h5'))
    except NO_SWA_ERROR:
        pass
    print(NAME.split('.h5')[0])


if __name__ == "__main__":
    main()
    