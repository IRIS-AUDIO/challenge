import numpy as np
import tensorflow as tf


def D_direction(y_true, y_pred):
    kernel = tf.constant([0.05, 0.1, 0.7, 0.1, 0.05], dtype=tf.float32)
    kernel = tf.reshape(kernel, (-1, 1, 1))

    wma_true = tf.nn.conv1d(tf.expand_dims(y_true, -1), kernel, 1, 'SAME')
    wma_pred = tf.nn.conv1d(tf.expand_dims(y_pred, -1), kernel, 1, 'SAME')

    return tf.reduce_sum(tf.square(wma_true - wma_pred), axis=(1, 2))


def D_class(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)

