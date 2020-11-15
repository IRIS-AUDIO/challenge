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
import tensorflow as tf

def main(config):
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
            
    with strategy.scope():
        model = getattr(models, 'model')(config).clf_model
        if config.mode == 'regr':
            model.layers[-1].activation = tf.keras.layers.ReLU()
        model.load_weights(NAME)
        # model = tf.keras.models.load_model(NAME)
        # train_set = make_dataset(config, training=True)
        test_set = make_dataset(config, training=False)
        

        for i,jj in test_set.take(10):
            pdb.set_trace()

if __name__ == "__main__":
    import sys
    main(get_args(sys.argv[1:]))