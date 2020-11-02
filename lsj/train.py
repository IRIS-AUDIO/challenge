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
        model = getattr(models, 'model')(config)
        
        train_set = make_dataset(config, training=True)
        test_set = make_dataset(config, training=False)


        for i, jj in train_set.take(1):
            pdb.set_trace()
            print(i.shape, jj.shape)

if __name__ == "__main__":
    import sys
    main(get_args(sys.argv[1:]))