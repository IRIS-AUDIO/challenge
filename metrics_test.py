import os
import numpy as np
import tensorflow as tf
from metrics import *


class MetricsTest(tf.test.TestCase):
    def setUp(self):
        self.gt = [[0, 0, 10], [2, 0, 20], [1, 15, 30], [2, 31, 40], [1, 27, 32]]
        self.predict = [[1, 5], [1, 19], [2, 32], [2, 38], [0, 38]]
        self.metric = Challenge_Metric()

    def test_get_er(self):
        self.assertEqual(1.2, get_er(self.gt, self.predict))

    def test_get_start_end_time(self):
        data = tf.cast([[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0]], tf.float32)
        res0 = tf.cast(tf.constant([[0, 0], [3, 4]]), tf.int64)
        res1 = tf.cast(tf.constant([[1, 1]]), tf.int64)
        res2 = tf.cast(tf.constant([[1, 3]]), tf.int64)

        cls0, cls1, cls2 = self.metric.get_start_end_time(data)
        self.assertEqual(tf.reduce_sum(tf.cast(res0 != cls0, tf.float32)), 0)
        self.assertEqual(tf.reduce_sum(tf.cast(res1 != cls1, tf.float32)), 0)
        self.assertEqual(tf.reduce_sum(tf.cast(res2 != cls2, tf.float32)), 0)
        
    # def test_gather_and_error(self):
    #     # pred = (batch, frame, classes)
    #     pred = tf.cast([[[0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0]]], tf.float32)
    #     true = tf.cast([[[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]]], tf.float32)
    #     self.assertEqual(1.2, self.metric.gather_and_error(true, pred))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

