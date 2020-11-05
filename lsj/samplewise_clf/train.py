import tensorflow as tf

import argparse, pdb
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from param import get_args
import models
from utils import *

    
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



def main(config):
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
    
    
    
    with strategy.scope():
        model = getattr(models, 'model')(config).clf_model
        
        train_set = make_dataset(config, training=True)
        test_set = make_dataset(config, training=False)
        

        if config.optimizer == 'adam':
            opt = Adam(config.lr) 
        elif config.optimizer == 'sgd':
            opt = SGD(config.lr, momentum=0.9)
        else:
            opt = RMSprop(config.lr, momentum=0.9)

        if config.l2 > 0:
            model = apply_kernel_regularizer(
                model, tf.keras.regularizers.l2(config.l2))

        if config.mode == 'regr':
            loss = 'MAE'
            model.layers[-1].activation = tf.keras.activations.relu
        elif config.mode == 'clf':
            loss = 'sparse_categorical_crossentropy'
        model.compile(optimizer=opt, 
                      loss=loss, # 'binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        
        """ TRAINING """
        callbacks = [
            # CSVLogger(NAME.replace('.h5', '.log'),
            #           append=True),
            LearningRateScheduler(custom_scheduler(config.n_dim*8, 
                                                   TOTAL_EPOCH/10)),
            # SWA(start_epoch=TOTAL_EPOCH//2, swa_freq=2),
            ModelCheckpoint(NAME,
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True),
            TerminateOnNaN(),
            TensorBoard(log_dir='tensorboard_log/'+config.name)
        ]

        model.fit(train_set,
                  epochs=config.epochs,
                  batch_size=config.batch_size,
                  steps_per_epoch=config.steps_per_epoch,
                  validation_data=test_set,
                  validation_steps=24,
                  callbacks=callbacks)

        # for i,jj in train_set.take(1):
        #     pdb.set_trace()

if __name__ == "__main__":
    import sys
    main(get_args(sys.argv[1:]))