import tensorflow as tf

import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from param import get_args
from models import *
import efficientnet.model as model


def main(config):
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
    N_CLASSES = 30

if __name__ == "__main__":
    import sys
    main(get_args(sys.argv[1:]))