import argparse
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

from metrics import *
from pipeline import *
from data_utils import *
from swa import SWA, NO_SWA_ERROR
from transforms import *
from utils import *


class ARGS:
    def __init__(self) -> None:
        self.args = argparse.ArgumentParser()
        self.args.add_argument('--name', type=str, default='')
        self.args.add_argument('--gpus', type=str, default='-1')
        self.args.add_argument('--model', type=int, default=0)
        self.args.add_argument('--model_type', type=str, default='vad', choices=['vad', 'eff', 'se'])
        self.args.add_argument('--v', type=int, default=1)
        self.args.add_argument('--pretrain', type=bool, default=False)
        self.args.add_argument('--n_layers', type=int, default=0)
        self.args.add_argument('--n_dim', type=int, default=256)
        self.args.add_argument('--n_chan', type=int, default=2)
        self.args.add_argument('--n_classes', type=int, default=3)
        self.args.add_argument('--patience', type=int, default=10)

        # DATA
        self.args.add_argument('--mse_multiplier', type=int, default=1)
        self.args.add_argument('--datapath', type=str, default='/root/datasets/Interspeech2020/generate_wavs/codes')
        self.args.add_argument('--background_sounds', type=str, default='drone_normed_complex_v4.pickle')
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

        self.args.add_argument('--epochs', type=int, default=300)
        self.args.add_argument('--batch_size', type=int, default=12)
        self.args.add_argument('--n_frame', type=int, default=512)
        self.args.add_argument('--steps_per_epoch', type=int, default=100)
        self.args.add_argument('--l1', type=float, default=0)
        self.args.add_argument('--l2', type=float, default=1e-6)
        self.args.add_argument('--loss', type=str, default='BCE')

        # AUGMENTATION
        self.args.add_argument('--snr', type=float, default=-20)
        self.args.add_argument('--max_voices', type=int, default=7)
        self.args.add_argument('--max_noises', type=int, default=2)

    def get(self):
        return self.args.parse_args()


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
                             min_ratio=1,
                             seperate_noise_voice=config.model_type == 'se' and config.v == 9)
    if config.model_type == 'se' and config.v == 9:
        # pipeline = pipeline.map(complex_to_magphase)
        pipeline = pipeline.map(speech_enhancement_preprocess)
        pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
        pipeline = pipeline.map(label_downsample(32))
        return pipeline.prefetch(AUTOTUNE)
    pipeline = pipeline.map(to_frame_labels)
    if training: 
        pipeline = pipeline.map(augment)
    if config.n_chan == 1:
        pipeline = pipeline.map(mono_chan)
    elif config.n_chan == 3:
        pipeline = pipeline.map(stereo_mono)
    elif config.n_chan > 3:
        pipeline = pipeline.map(random_merge_aug(config.n_chan))
    if 'filter' in config.name:
        pipeline = pipeline.map(stft_filter(int(round(200 / (16000 / 256)))))
    pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    if 'nominmax' not in config.name:
        pipeline = pipeline.map(minmax)
    pipeline = pipeline.map(log_on_mel)
    if config.v in label_downsample_model:
        pipeline = pipeline.map(label_downsample(32))
    elif config.v == 5:
        pipeline = pipeline.map(label_downsample(config.n_frame // (config.n_frame * 256 // 16000)))
    if config.loss.upper() in ('MSE', 'MAE'):
        pipeline = pipeline.map(multiply_label(config.mse_multiplier))
    return pipeline.prefetch(AUTOTUNE)


def custom_scheduler(d_model, warmup_steps=4000, lr_div=2):
    # https://www.tensorflow.org/tutorials/text/transformer#optimizer
    d_model = tf.cast(d_model, tf.float32)

    def _scheduler(step):
        step = tf.cast(step+1, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (warmup_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2) / lr_div
    return _scheduler


def adaptive_clip_grad(parameters, gradients, clip_factor=0.01,
                       eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads


class CustomModel(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super(CustomModel, self).__init__(**kwargs)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        if not isinstance(y, tuple):
            y = (y,)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            if not isinstance(y_pred, (tuple, list)):
                y_pred = (y_pred,)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        
        gradients = tape.gradient(loss, trainable_vars)
        gradients = adaptive_clip_grad(self.trainable_variables, gradients)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        
        self.compiled_metrics.update_state(y, y_pred[0])

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def ConvMPBlock(x, num_convs=2, fsize=32, kernel_size=3, pool_size=(2,2), strides=(2,2), BN=False, DO=False, MP=True):
    for i in range(num_convs):
       x = tf.keras.layers.Conv2D(fsize, kernel_size, padding='same')(x)
       if BN:
           x = tf.keras.layers.BatchNormalization()(x)
       if DO:
           x = tf.keras.layers.Dropout(DO)(x)
       x = tf.keras.layers.Activation('relu')(x)
    if MP:
        x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding='same')(x)
    return x


def FullyConnectedLayer(x, nodes=512, act='relu', BN=False, DO=False, name=None):
    x = tf.keras.layers.Dense(nodes)(x)
    if BN:
        x = tf.keras.layers.BatchNormalization()(x)
    if DO:
        x = tf.keras.layers.Dropout(DO)(x)
    x = tf.keras.layers.Activation(act, name=name)(x)
    return x


def define_keras_model(config=None):
    fsize = 32
    if config.model_type == 'vad' and config.v == 8:
        fsize = 48

    td_dim = 1024
    input_tensor = tf.keras.layers.Input(
        shape=(config.n_mels, config.n_frame, config.n_chan))
    x = input_tensor
    x = ConvMPBlock(x, num_convs=2, fsize=fsize, BN=True)
    for i in range(1, 5):
        if config.model_type == 'vad' and config.v == 6:
            seconds = 0.5
            kernel_size = int(round(seconds / (256 * config.n_frame / 16000 / x.shape[-2])))
            x = tf.keras.layers.AveragePooling2D((1,kernel_size,), 1, padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((1,kernel_size * 2,), 1, padding='same')(x)
        if config.model_type == 'vad' and config.v == 7:
            skip = x
            x = tf.keras.layers.Conv2D(skip.shape[-1] // 4, 1, 1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(skip.shape[-1] // 4, 3, 1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(skip.shape[-1], 1, 1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x += skip
        x = ConvMPBlock(x, num_convs=3, fsize=fsize * 2**i, BN=True)

    x = tf.keras.layers.Permute((2,1,3))(x)
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(td_dim, activation='relu'))(x)
    if config.model_type == 'vad' and config.v == 9:
        x = FullyConnectedLayer(x, 512, BN=True)
    x = FullyConnectedLayer(x, 256, BN=True)
    x = FullyConnectedLayer(x, 128, BN=True)
    if config.model_type == 'vad' and config.v == 9:
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = FullyConnectedLayer(x, 64, BN=True)
    x = FullyConnectedLayer(x, 3, act='sigmoid' if config.model_type == 'vad' else 'relu', name='class' if config.model_type == 'se' else None)
    return CustomModel(inputs=input_tensor, outputs=x)


def convset(inp, chan=16):
    out = inp
    for _ in range(2):
        out = tf.keras.layers.Conv2D(chan, 3, strides=1, padding='same')(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.MaxPooling2D()(out)
    return out


def upsampling(inp, chan=64):
    out = inp
    out = tf.keras.layers.Conv2D(chan, 3, strides=1, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    return tf.keras.layers.Conv2DTranspose(chan, 2, 2, padding='same')(out)


def speech_enhancement_model(input):
    merge_input = tf.keras.layers.Input(tensor=input[1:])
    inp1 = convset(merge_input, 64)
    inp2 = convset(inp1, 128)
    inp3 = convset(inp2, 256)
    latent = convset(inp3, 512)
    
    speech3 = upsampling(latent, 256)
    speech2 = upsampling(tf.keras.layers.Concatenate(-1)([inp3, speech3]), 128)
    speech1 = upsampling(tf.keras.layers.Concatenate(-1)([inp2, speech2]), 64)
    speech = upsampling(tf.keras.layers.Concatenate(-1)([inp1, speech1]), 2)

    noise3 = upsampling(latent, 256)
    noise2 = upsampling(tf.keras.layers.Concatenate(-1)([inp3, noise3]), 128)
    noise1 = upsampling(tf.keras.layers.Concatenate(-1)([inp2, noise2]), 64)
    noise = upsampling(tf.keras.layers.Concatenate(-1)([inp1, noise1]), 2)
    return CustomModel(inputs=merge_input, outputs=[speech, noise])


def get_model(config):
    input_tensor = tf.keras.layers.Input(
        shape=(config.n_mels, config.n_frame, config.n_chan))

    if config.model_type == 'se':
        input_tensor = tf.keras.layers.Input(shape=(256, config.n_frame, config.n_chan))
        merge_input = input_tensor[:, 1:]
        merge_input = tf.transpose(input_tensor, perm=[0, 2, 1, 3])

        se_model = speech_enhancement_model(merge_input)
        if not config.pretrain:
            se_model.trainable = False
        speech, noise = se_model(merge_input)
        
        # out = tf.keras.layers.Concatenate(-1)([speech, noise])
        out = speech
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        config.n_mels = out.shape[1]
        tmp_config = deepcopy(config)
        tmp_config.n_chan = out.shape[-1]
        vadmodel = define_keras_model(tmp_config)
        if config.pretrain:
            vadmodel.trainable = False
        out = vadmodel(out)

        # backbone = getattr(tf.keras.applications.efficientnet, f'EfficientNetB4')(
        # include_top=False, weights=None, input_tensor=out)
        # out = tf.keras.layers.Permute((2, 1, 3))(backbone.output)
        # out = tf.keras.layers.Reshape((-1, out.shape[-1] * out.shape[-2]))(out)
        # out = tf.keras.layers.Conv1DTranspose(128, 2, 2)(out)
        # out = tf.keras.layers.Activation('relu')(out)
        # out = tf.keras.layers.Conv1DTranspose(64, 2, 2)(out)
        # out = tf.keras.layers.Activation('relu')(out)
        # out = tf.keras.layers.Conv1DTranspose(32, 2, 2)(out)
        # out = tf.keras.layers.Activation('relu')(out)
        # out = tf.keras.layers.Conv1DTranspose(16, 2, 2)(out)
        # out = tf.keras.layers.Activation('relu')(out)
        # out = tf.keras.layers.Conv1DTranspose(8, 2, 2)(out)
        # out = tf.keras.layers.Activation('relu')(out)
        # out = tf.keras.layers.Dense(config.n_classes)(out)
        # out = tf.keras.layers.Activation('sigmoid', name='class')(out)

        speech = tf.keras.layers.Permute((2, 1, 3), name='speech')(speech)
        noise = tf.keras.layers.Permute((2, 1, 3), name='noise')(noise)
        return CustomModel(inputs=[input_tensor], outputs=[out, speech, noise])
    elif config.model_type == 'eff':
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
            raise ValueError('version 2 is deprecated')
            out = tf.keras.layers.Conv1DTranspose(128, 2, 2)(out)
            out = tf.keras.layers.Conv1DTranspose(64, 2, 2)(out)
            out = tf.keras.layers.Conv1DTranspose(32, 2, 2)(out)
            out = tf.keras.layers.Conv1DTranspose(16, 2, 2)(out)
            out = tf.keras.layers.Conv1DTranspose(3, 2, 2)(out)
        elif config.v == 3:
            out = out
        elif config.v == 4:
            raise ValueError('version 4 is deprecated')
            out = tf.keras.layers.Conv1D(config.n_frame, 1, use_bias=False, data_format='channels_first')(out)
        elif config.v == 5:
            if out.shape[1] != config.n_frame * 256 // 16000:
                out = tf.keras.layers.Conv1D(config.n_frame * 256 // 16000, 1, use_bias=False, data_format='channels_first')(out)
                out = tf.keras.layers.BatchNormalization()(out)
                out = tf.keras.layers.Activation('relu')(out)
            out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(out)
        elif config.v == 6:
            out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(out)
            out = FullyConnectedLayer(out, 256, BN=True)
            out = FullyConnectedLayer(out, 128, BN=True)
            out = FullyConnectedLayer(out, 64, BN=True)
        elif config.v == 7:
            out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(out)
            big = tf.keras.layers.Reshape((config.n_mels, -1))(input_tensor)
            big = tf.keras.layers.Conv1D(out.shape[-1], 16, strides=5, padding='same')(big)
            big = tf.keras.layers.Activation('tanh')(big)
            out *= big
        else:
            raise ValueError('wrong version')
            
        out = tf.keras.layers.Dense(config.n_classes)(out)
        # out= tf.keras.layers.Activation('relu')(out)
        # out *= tf.cast(out < 1., out.dtype)
        out = tf.keras.layers.Activation('sigmoid')(out)
        return tf.keras.models.Model(inputs=input_tensor, outputs=out)
    elif config.model_type == 'vad':
        return define_keras_model(config)


def main():
    config = ARGS().get()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    config.loss = config.loss.upper()
    if config.loss != 'MSE':
        config.mse_multiplier = 1
    print(config)

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = (config.name + '_') if config.name != '' else ''
    if config.model_type == 'eff':
        model_first_name = f'B{config.model}'
    elif config.model_type == 'se':
        model_first_name = 'se'
    elif config.model_type == 'vad':
        model_first_name = 'vad'
        
    NAME = NAME + '_'.join([model_first_name, f'v{config.v}', f'lr{config.lr}', 
                            f'batch{config.batch_size}', f'opt_{config.optimizer}', 
                            f'mel{config.n_mels}', f'chan{config.n_chan}', f'{config.loss.upper()}', f'framelen{config.n_frame}'])
    if config.model_type == 'se' and config.v == 9 and config.pretrain:
        NAME += '_weight'
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
        raise ValueError('adabelief is deprecated')
        opt = AdaBelief(lr, clipvalue=config.clipvalue)
    # if config.l2 > 0:
    #     model = apply_kernel_regularizer(
    #         model, tf.keras.regularizers.l1_l2(config.l1, config.l2))

    if config.loss.upper() == 'BCE':
        loss = tf.keras.losses.BinaryCrossentropy()
    elif config.loss.upper() == 'FOCAL':
        loss = sigmoid_focal_crossentropy
    if config.model_type == 'se' and config.v == 9:
        loss = [loss, tf.losses.MAE, tf.losses.MAE]

    metrics = [cos_sim,
               f1_score()]
    if config.v != 5:
        metrics.append(er_score(smoothing=False))
    model.compile(optimizer=opt, 
                #   loss=custom_loss(alpha=config.loss_alpha, l2=config.loss_l2),
                  loss=loss,
                  loss_weights=[1, 10, 10],
                  metrics=metrics)
    setattr(model, 'train_config', config)
    model.summary()
    print(NAME)

    if config.model_type == 'se' and config.v == 9 and not config.pretrain:
        model.load_weights(NAME)
        print('loaded pretrained model')

    """ DATA """
    train_set = make_dataset(config, training=True)
    test_set = make_dataset(config, training=False)
    
    earlystop_monitor = 'val_loss'
    model_checkpoint_monitor = 'val_class_er' if config.v == 9 and config.model_type == 'eff' else 'val_er'
    if config.model_type == 'se' and config.v == 9:
        if config.pretrain:
            earlystop_monitor = 'val_speech_loss'
            model_checkpoint_monitor = 'val_speech_loss'
        else:
            earlystop_monitor = 'val_class_loss'
            model_checkpoint_monitor = 'val_class_er'
    else:
        earlystop_monitor = 'val_loss'
        model_checkpoint_monitor = 'val_er'

    """ TRAINING """
    callbacks = [
        CSVLogger(NAME.replace('.h5', '.csv'), append=True),
        SWA(start_epoch=TOTAL_EPOCH//4, swa_freq=2),
        ModelCheckpoint(NAME, monitor=model_checkpoint_monitor, save_best_only=True, verbose=1),
        TerminateOnNaN(),
        TensorBoard(log_dir=os.path.join('tensorboard_log', NAME.split('.h5')[0])),
        EarlyStopping(monitor=earlystop_monitor, patience=config.patience, restore_best_weights=True),
        eval_callback(config, NAME),
        # LearningRateScheduler(tf.keras.optimizers.schedules.CosineDecayRestarts(config.lr, 5), verbose=1),
        # LearningRateScheduler(lr_schedule, verbose=1),
        # ReduceLROnPlateau(monitor='val_loss', factor=1 / 2**0.5, patience=5, verbose=1, mode='min')
    ]
    callbacks.append(
        LearningRateScheduler(
            custom_scheduler(4096, TOTAL_EPOCH/12, config.lr_div)))

    # if not config.pretrain:
    #     callbacks.append(
    #         LearningRateScheduler(
    #             custom_scheduler(4096, TOTAL_EPOCH/12, config.lr_div)))
    # else:
    #     callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=1 / 2**0.5, patience=5, verbose=1, mode='min'))

    try:
        model.fit(train_set,
                epochs=TOTAL_EPOCH,
                batch_size=BATCH_SIZE,
                steps_per_epoch=config.steps_per_epoch,
                validation_data=test_set,
                validation_steps=16,
                callbacks=callbacks)
        print('best model:', NAME.replace('.h5', '_SWA.h5'))
        model.save(NAME.replace('.h5', '_SWA.h5'))
    except NO_SWA_ERROR:
        pass
    print(NAME.split('.h5')[0])
    exit()


if __name__ == "__main__":
    main()
    
