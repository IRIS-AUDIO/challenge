import argparse
import numpy as np
import os
import pickle, pdb
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
from models import transformer_layer
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--pretrain', type=bool, default=False)
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dim', type=int, default=256)
args.add_argument('--n_heads', type=int, default=8)

abspath = '/root/otherperson/daniel'
# DATA
args.add_argument('--background_sounds', type=str,
                  default=abspath+'/generate_wavs/drone_normed_complex_v3.pickle')
args.add_argument('--voices', type=str,
                  default=abspath+'/generate_wavs/voice_normed_complex_v2_2.pickle')
args.add_argument('--labels', type=str,
                  default=abspath+'/generate_wavs/voice_labels_mfc_v2_1.npy')
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
args.add_argument('--lr', type=float, default=0.001)

args.add_argument('--epochs', type=int, default=500)
args.add_argument('--batch_size', type=int, default=12)
args.add_argument('--n_frame', type=int, default=2048)
args.add_argument('--steps_per_epoch', type=int, default=100)
args.add_argument('--l2', type=float, default=1e-6)
args.add_argument('--loss_l2', type=float, default=1.)
args.add_argument('--multiplier', type=float, default=10)
args.add_argument('--filter', type=int, default=0)


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
    specs = mask(specs, axis=time_axis, max_mask_size=24, n_mask=6) 
    specs = mask(specs, axis=freq_axis, max_mask_size=16)
    specs, labels = random_reverse_chan(specs, labels)
    return specs, labels


def preprocess_labels(multiplier):
    def _preprocess(x, y):
        # process y: [None, time, classes] -> [None, time', classes]
        for i in range(5):
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
                             config=config)
    pipeline = pipeline.map(to_density_labels)
    if training: 
        pipeline = pipeline.map(augment)
    pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    pipeline = pipeline.map(minmax_log_on_mel)
    pipeline = pipeline.map(preprocess_labels(config.multiplier))
    return pipeline.prefetch(AUTOTUNE)


def Doa_loss(alpha=1, l2=1):
    def _custom(y_true, y_pred):
        # opt alpha 0.5 (bad) 0.8 (best) 0.7 (bad)
        # optimal l2 -> 0.5 (no), 2 (no), 1 (best) 
        # y_true, y_pred = [None, time, 30]
        # [None, time, 30] -> [None, time, 3, 10]
        t_true = tf.stack(tf.split(y_true, 3, axis=-1), axis=-2)
        # t_pred = tf.stack(tf.split(y_pred, 3, axis=-1), axis=-2)
        t_pred = y_pred

        # [None, time, 10]
        d_y_true = tf.reduce_sum(t_true, axis=-2)
        # d_y_pred = tf.reduce_sum(t_pred, axis=-2)
        d_y_pred = t_pred

        loss = alpha * tf.keras.losses.MAE(tf.reduce_sum(d_y_true, axis=1),
                                           tf.reduce_sum(d_y_pred, axis=1))

        # TODO: OT loss

        # TV: total variation loss
        # normed - degrees [None, time, 10]
        n_d_true = safe_div(
            d_y_true, tf.reduce_sum(d_y_true, axis=1, keepdims=True))
        n_d_pred = safe_div(
            d_y_pred, tf.reduce_sum(d_y_pred, axis=1, keepdims=True))


        tv = alpha * tf.reduce_mean(
                tf.reduce_sum(tf.math.abs(n_d_true - n_d_pred), axis=1) 
                * tf.reduce_sum(d_y_true, axis=1), # [None, 10]
                axis=1)
        loss += l2 * tv

        return loss
    return _custom


def Numbering_loss(alpha=0, l2=1):
    def _custom(y_true, y_pred):
        # opt alpha 0.5 (bad) 0.8 (best) 0.7 (bad)
        # optimal l2 -> 0.5 (no), 2 (no), 1 (best) 
        # y_true, y_pred = [None, time, 30]
        # [None, time, 30] -> [None, time, 3, 10]
        t_true = tf.stack(tf.split(y_true, 3, axis=-1), axis=-2)
        # t_pred = tf.stack(tf.split(y_pred, 3, axis=-1), axis=-2)
        t_pred = y_pred

        # [None, time, 3]
        c_y_true = tf.reduce_sum(t_true, axis=-1)
        # c_y_pred = tf.reduce_sum(t_pred, axis=-1)
        c_y_pred = t_pred

        loss = (1-alpha) * tf.keras.losses.MAE(tf.reduce_sum(c_y_true, axis=1),
                                               tf.reduce_sum(c_y_pred, axis=1))


        # normed - classes [None, time, 3]
        n_c_true = safe_div(
            c_y_true, tf.reduce_sum(c_y_true, axis=1, keepdims=True))
        n_c_pred = safe_div(
            c_y_pred, tf.reduce_sum(c_y_pred, axis=1, keepdims=True))
        tv = (1-alpha) * tf.reduce_mean(
                tf.reduce_sum(tf.math.abs(n_c_true - n_c_pred), axis=1) 
                * tf.reduce_sum(c_y_true, axis=1), # [None, 3]
                axis=1)
        loss += l2 * tv

        return loss
    return _custom


def custom_scheduler(d_model, warmup_steps=4000):
    # will be replaced by exponential scheduler
    # https://www.tensorflow.org/tutorials/text/transformer#optimizer
    d_model = tf.cast(d_model, tf.float32)

    def _scheduler(step):
        step = tf.cast(step+1, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (warmup_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2) # / 2. # TEST
    return _scheduler

def get_model(config):
    x = tf.keras.layers.Input(shape=(config.n_mels, config.n_frame, 2))
    first_model = getattr(model, config.model)(
        include_top=False,
        weights=None,
        input_tensor=x,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils,
    )
    out = tf.transpose(first_model.output, perm=[0, 2, 1, 3])
    out = tf.keras.layers.Reshape([-1, out.shape[-1]*out.shape[-2]])(out)
    if config.n_layers > 0:
        out = tf.keras.layers.Dense(config.n_dim)(out)
        for i in range(config.n_layers):
            out = transformer_layer(config.n_dim, config.n_heads)(out)

    out = tf.keras.layers.Dense(3, activation='relu')(out)
    first_model = tf.keras.models.Model(inputs=first_model.input, outputs=out)

    x = tf.keras.layers.Input(shape=(config.n_mels, config.n_frame, 5))
    second_model = getattr(model, config.model)(
        include_top=False,
        weights=None,
        input_tensor=x,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils,
    )
    out = tf.transpose(second_model.output, perm=[0, 2, 1, 3])
    out = tf.keras.layers.Reshape([-1, out.shape[-1]*out.shape[-2]])(out)

    if config.n_layers > 0:
        out = tf.keras.layers.Dense(config.n_dim)(out)
        for i in range(config.n_layers):
            out = transformer_layer(config.n_dim, config.n_heads)(out)

    out = tf.keras.layers.Dense(10, activation='relu')(out)
    second_model = tf.keras.models.Model(inputs=second_model.input, outputs=out)

    if config.l2 > 0:
        first_model = apply_kernel_regularizer(
            first_model, tf.keras.regularizers.l2(config.l2))
        second_model = apply_kernel_regularizer(
            second_model, tf.keras.regularizers.l2(config.l2))

    return first_model, second_model

@tf.function
def cos_sim(y_true, y_pred):
    mask = tf.cast(
        tf.reduce_sum(y_true, axis=-2) > 0., tf.float32) # [None, 30]
    mask = safe_div(mask, tf.reduce_sum(mask, axis=-1, keepdims=True))
    return tf.reduce_sum(
        tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-2) * mask, 
        axis=-1)

# @tf.function
def d_num_total(y_true, y_pred, apply_round=True, multiplier=10):
    y_true /= multiplier
    y_pred /= multiplier

    # [None, time, 30] -> [None, time, 3, 10]
    y_true = tf.stack(tf.split(y_true, 3, axis=-1), axis=-2)
    y_pred = y_pred

    pdb.set_trace()
    # c_cls
    c_true = tf.reduce_sum(y_true, axis=(-3, -2))
    c_pred = tf.reduce_sum(y_pred, axis=-2)
    if apply_round:
        c_true = tf.math.round(c_true)
        c_pred = tf.math.round(c_pred)
    d_cls = D_class(c_true, c_pred)

    return 0.2 * d_cls

@tf.function
def d_doa_total(y_true, y_pred, apply_round=True, multiplier=10):
    y_true /= multiplier
    y_pred /= multiplier

    # [None, time, 30] -> [None, time, 3, 10]
    y_true = tf.stack(tf.split(y_true, 3, axis=-1), axis=-2)
    y_pred = y_pred # [None, time, 10]

    pdb.set_trace()
    # d_dir
    d_true = tf.reduce_sum(y_true, axis=(-3, -2))
    d_pred = tf.reduce_sum(y_pred, axis=-2)
    if apply_round:
        d_true = tf.math.round(d_true)
        d_pred = tf.math.round(d_pred)
    d_dir = D_direction(d_true, d_pred)

    return 0.8 * d_dir

@tf.function
def train_step(model, loss_fn, x, y, train=True):
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x, training=train)
        loss_val = loss_fn(y, logits)
    _y = tf.reshape(y, (y.shape[0], y.shape[1], 3, 10))
    if logits.shape[-1] == 3:
        _y = tf.reduce_sum(_y, -1)
    if logits.shape[-1] == 10:
        _y = tf.reduce_sum(_y, -2)
    cos_val = cos_sim(_y, logits)
    grads = tape.gradient(loss_val, model.trainable_weights)
    logits = logits / config.multiplier
    num_res = tf.round(tf.reduce_sum(logits, axis=1))
    return loss_val, num_res, cos_val, grads


def train(first_model, second_model, trainset, testset, opt_num, opt_do, config):
    num_loss = Numbering_loss(alpha=0.8, l2=config.loss_l2)
    doa_loss = Doa_loss(alpha=0.8, l2=config.loss_l2)
    best_score = 10000
    tensorboard_dir = 'tensorboard_log/' + config.name + '/train'
    model_dir = 'model_save/' + config.name
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    train_num_loss = tf.keras.metrics.Mean('train_num_loss', dtype=tf.float32)
    train_doa_loss = tf.keras.metrics.Mean('train_doa_loss', dtype=tf.float32)
    train_num_cos = tf.keras.metrics.Mean('train_num_cos', dtype=tf.float32)
    train_doa_cos = tf.keras.metrics.Mean('train_doa_cos', dtype=tf.float32)
    train_d_total = tf.keras.metrics.Mean('train_d_total', dtype=tf.float32)
    val_num_loss = tf.keras.metrics.Mean('val_num_loss', dtype=tf.float32)
    val_doa_loss = tf.keras.metrics.Mean('val_doa_loss', dtype=tf.float32)
    val_num_cos = tf.keras.metrics.Mean('val_num_cos', dtype=tf.float32)
    val_doa_cos = tf.keras.metrics.Mean('val_doa_cos', dtype=tf.float32)
    val_d_total = tf.keras.metrics.Mean('val_d_total', dtype=tf.float32)

    for epoch in range(config.epochs):
        with tqdm(train_set) as pbar:
            for step, (x_batch, Y_batch) in enumerate(pbar):
                num_y_batch = Y_batch
                doa_y_batch = Y_batch

                num_lossval, num_logits, num_cos, num_grads = train_step(first_model, num_loss, x_batch, num_y_batch, train=True)
                opt_num.apply_gradients(zip(num_grads, first_model.trainable_weights))
                score = d_num_total(Y_batch, num_logits)

                numbers = tf.reshape(tf.repeat(num_logits, x_batch.shape[1] * x_batch.shape[2], 0), (x_batch.shape[0], x_batch.shape[1], x_batch.shape[2],3))
                doa_lossval, doa_logits, doa_cos, doa_grads = train_step(second_model, doa_loss, tf.concat([x_batch, numbers], -1), doa_y_batch, train=True)
                opt_do.apply_gradients(zip(doa_grads, second_model.trainable_weights))
                score += d_doa_total(Y_batch, doa_logits)
                
                train_num_loss.update_state(num_lossval)
                train_doa_loss.update_state(doa_lossval)
                train_num_cos.update_state(num_cos)
                train_doa_cos.update_state(doa_cos)
                train_d_total.update_state(score)
                
                if step == config.steps_per_epoch:
                    break
                pbar.set_postfix(epoch=f'{epoch:3}')

        with summary_writer.as_default():
            tf.summary.scalar('num_loss', train_num_loss.result(), step=epoch)
            tf.summary.scalar('doa_loss', train_doa_loss.result(), step=epoch)
            tf.summary.scalar('num_cos', train_num_cos.result(), step=epoch)
            tf.summary.scalar('doa_cos', train_doa_cos.result(), step=epoch)
            tf.summary.scalar('d_total', train_d_total.result(), step=epoch)

        with tqdm(test_set) as pbar:
            for step, (x_batch, Y_batch) in enumerate(pbar):
                num_y_batch = Y_batch
                doa_y_batch = Y_batch

                num_lossval, num_logits, num_cos, num_grads = train_step(first_model, num_loss, x_batch, num_y_batch, train=False)
                score = d_num_total(Y_batch, num_logits)
                numbers = tf.reshape(tf.repeat(num_logits, x_batch.shape[1] * x_batch.shape[2], 0), (x_batch.shape[0], x_batch.shape[1], x_batch.shape[2],3))
                doa_lossval, doa_logits, doa_cos, doa_grads = train_step(second_model, doa_loss, tf.concat([x_batch, numbers], -1), doa_y_batch, train=False)
                score += d_doa_total(Y_batch, doa_logits)
                
                val_num_loss.update_state(num_lossval)
                val_doa_loss.update_state(doa_lossval)
                val_num_cos.update_state(num_cos)
                val_doa_cos.update_state(doa_cos)
                val_d_total.update_state(score)

                if step == config.steps_per_epoch:
                    break
                pbar.set_postfix(epoch=f'val{epoch:3}')

        with summary_writer.as_default():
            tf.summary.scalar('num_loss', val_num_loss.result(), step=epoch)
            tf.summary.scalar('doa_loss', val_doa_loss.result(), step=epoch)
            tf.summary.scalar('num_cos', val_num_cos.result(), step=epoch)
            tf.summary.scalar('doa_cos', val_doa_cos.result(), step=epoch)
            tf.summary.scalar('d_total', val_d_total.result(), step=epoch)

        if val_d_total.result() <= best_score:
            first_model.save(model_dir+'_first.h5')
            second_model.save(model_dir+'_second.h5')
            best_score = val_d_total.result()

        train_num_loss.reset_states()
        train_doa_loss.reset_states()
        train_num_cos.reset_states()
        train_doa_cos.reset_states()
        val_num_loss.reset_states()
        val_doa_loss.reset_states()
        val_num_cos.reset_states()
        val_doa_cos.reset_states()


        



if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'

    """ MODEL """
   

    
    # tf.reshape(tf.repeat(a,128*2048,0), (32,128,2048,3))


    lr = config.lr
    if config.optimizer == 'adam':
        opt_num = Adam(lr, clipvalue=0.01) 
        opt_do = Adam(lr, clipvalue=0.01) 
    elif config.optimizer == 'sgd':
        opt_num = SGD(lr, momentum=0.9)
        opt_do = SGD(lr, momentum=0.9)
    else:
        opt_num = RMSprop(lr, momentum=0.9)
        opt_do = RMSprop(lr, momentum=0.9)
    
    first_model, second_model = get_model(config)
    
    # model.summary()

    if config.pretrain:
        first_model.load_weights(NAME[:-3]+'first' + NAME[-3:])
        second_model.load_weights(NAME[:-3]+'second' + NAME[-3:])
        print('loaded pretrained model')

    """ DATA """
    train_set = make_dataset(config, training=True)
    test_set = make_dataset(config, training=False)

    """ TRAINING """
    callbacks = [
        CSVLogger(NAME.replace('.h5', '.log'),
                  append=True),
        LearningRateScheduler(custom_scheduler(4096, TOTAL_EPOCH/12)),
        SWA(start_epoch=TOTAL_EPOCH//2, swa_freq=2),
        ModelCheckpoint(NAME,
                        monitor='val_d_total', 
                        save_best_only=True,
                        verbose=1),
        TerminateOnNaN()
    ]

    train(first_model, second_model, train_set, test_set, opt_num, opt_do, config)
    # model.fit(train_set,
    #           epochs=TOTAL_EPOCH,
    #           batch_size=BATCH_SIZE,
    #           steps_per_epoch=config.steps_per_epoch,
    #           validation_data=test_set,
    #           validation_steps=12,
    #           callbacks=callbacks)

    model.save(NAME.replace('.h5', '_SWA.h5'))
