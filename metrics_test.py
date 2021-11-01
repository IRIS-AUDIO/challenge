import os
import numpy as np
import tensorflow as tf
from metrics import *


class MetricsTest(tf.test.TestCase):
    def setUp(self):
        self.gt = tf.convert_to_tensor([[0, 0, 10], [2, 0, 20], [1, 15, 30], [2, 31, 40], [1, 32, 35]])
        self.predict = tf.convert_to_tensor([[1, 5], [1, 19], [2, 32], [2, 38], [0, 38]])

    def test_get_er(self):
        self.assertEqual(1.2, get_er(self.gt, self.predict))

    def test_er_score(self):
        gt_numpy = self.gt.numpy()
        gt_array = np.zeros([2, 40, 3])
        pred_array = np.zeros([2, 40, 3])
        for item in gt_numpy:
            gt_array[0, item[1]:item[2], item[0]] = 1
            gt_array[1, item[1]:item[2], item[0]] = 1
        for item in self.predict.numpy():
            pred_array[0, item[1]-2:item[1]+2, item[0]] = 1
            pred_array[1, item[1]-2:item[1]+2, item[0]] = 1

        er_func = er_score(smoothing=False)
        er = er_func(gt_array, pred_array)
        self.assertEqual(tf.reduce_mean(er), 1.2)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

