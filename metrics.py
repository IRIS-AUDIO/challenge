import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
from utils import *


def er_score(threshold=0.5, smoothing=True):
    threshold = tf.constant(threshold, tf.float32)

    def er(y_true, y_pred):
        y_true = tf.cast(y_true >= threshold, tf.int32)
        if smoothing:
            smoothing_kernel_size = int(0.5 * 16000) // 256 # 0.5
            y_pred = tf.keras.layers.AveragePooling1D(smoothing_kernel_size, padding='same')(y_pred)
        y_pred = tf.cast(y_pred >= threshold, tf.int32)

        # True values
        # [batch, time, cls]
        true_starts = tf.clip_by_value(
            y_true - tf.pad(y_true, [[0, 0], [1, 0], [0, 0]])[:, :-1], 0, 1)
        true_ends = tf.clip_by_value(
            y_true - tf.pad(y_true, [[0, 0], [0, 1], [0, 0]])[:, 1:], 0, 1)
        n_true = tf.reduce_sum(tf.cast(true_starts, tf.float32), (1, 2))

        true_starts = tf.where(true_starts)
        true_ends = tf.where(true_ends)
        true_starts = tf.gather(true_starts, tf.argsort(true_starts[:, -1]), -1)
        true_starts = tf.gather(true_starts, tf.argsort(true_starts[:, 0]), 0)
        true_ends = tf.gather(true_ends, tf.argsort(true_ends[:, -1]), -1)
        true_ends = tf.gather(true_ends, tf.argsort(true_ends[:, 0]), 0)

        # prediction values
        pred_starts = tf.clip_by_value(
            y_pred - tf.pad(y_pred, [[0, 0], [1, 0], [0, 0]])[:, :-1], 0, 1)
        pred_ends = tf.clip_by_value(
            y_pred - tf.pad(y_pred, [[0, 0], [0, 1], [0, 0]])[:, 1:], 0, 1)
        n_pred = tf.reduce_sum(tf.cast(pred_starts, tf.float32), (1, 2))

        pred_starts = tf.where(pred_starts)
        pred_ends = tf.where(pred_ends)
        pred_starts = tf.gather(pred_starts, tf.argsort(pred_starts[:, -1]), -1)
        pred_starts = tf.gather(pred_starts, tf.argsort(pred_starts[:, 0]), 0)
        pred_ends = tf.gather(pred_ends, tf.argsort(pred_ends[:, -1]), -1)
        pred_ends = tf.gather(pred_ends, tf.argsort(pred_ends[:, 0]), 0)

        middle = tf.cast((pred_starts+pred_ends)/2, tf.int64)

        # correct: correct batch and cls (true, pred)
        correct = (
            true_starts[:, ::2, None]==tf.transpose(middle, (1, 0))[None, ::2])
        correct = tf.reduce_min(tf.cast(correct, tf.float32), axis=1)

        mid_time = tf.transpose(middle[:, 1:2], (1, 0))
        correct *= tf.cast(true_starts[:, 1:2] <= mid_time, tf.float32)
        correct *= tf.cast(true_ends[:, 1:2] >= mid_time, tf.float32)
        correct = tf.reduce_max(tf.pad(correct, [[0, 0], [0, 1]]), -1)

        correct_per_sample = tf.reduce_sum(
            tf.one_hot(true_starts[:, 0], tf.shape(y_pred)[0])*correct[:, None],
            0)
        score = n_true + n_pred - 2 * correct_per_sample
        score /= tf.clip_by_value(n_true, 1, tf.reduce_max(n_true))
        return score
    return er


def cos_sim(y_true, y_pred):
    mask = tf.cast(
        tf.reduce_sum(y_true, axis=-2) > 0., tf.float32) # [None, 3]
    mask = safe_div(mask, tf.reduce_sum(mask, axis=-1, keepdims=True))
    return tf.reduce_sum(
        tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-2) * mask, 
        axis=-1)

