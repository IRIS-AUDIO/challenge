import os
import numpy as np
import tensorflow as tf
from metrics import *


class MetricsTest(tf.test.TestCase):
    def setUp(self):
        self.gt = tf.convert_to_tensor([[0, 0, 10], [2, 0, 20], [1, 15, 30], [2, 31, 40], [1, 32, 35]])
        self.predict = tf.convert_to_tensor([[1, 5], [1, 19], [2, 32], [2, 38], [0, 38]])
        self.metric = Challenge_Metric()

    def test_get_er(self):
        self.assertEqual(1.2, get_er(self.gt, self.predict))

    def test_get_start_end_frame(self):
        data = tf.cast([[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0]], tf.float32)
        res0 = tf.cast(tf.constant([[0, 0], [3, 4]]), tf.int64)
        res1 = tf.cast(tf.constant([[1, 1]]), tf.int64)
        res2 = tf.cast(tf.constant([[1, 3]]), tf.int64)

        cls0, cls1, cls2 = self.metric.get_start_end_frame(data)
        self.assertEqual(tf.reduce_sum(tf.cast(res0 != cls0, tf.float32)), 0)
        self.assertEqual(tf.reduce_sum(tf.cast(res1 != cls1, tf.float32)), 0)
        self.assertEqual(tf.reduce_sum(tf.cast(res2 != cls2, tf.float32)), 0)

    def test_second_answer(self):
        data = tf.random.uniform([450, 3])
        answer = self.metric.get_second_answer(data)

    def test_tf_er(self):
        gt_numpy = self.gt.numpy()
        gt_array = np.zeros([2, 40, 3])
        pred_array = np.zeros([2, 40, 3])
        for item in gt_numpy:
            gt_array[0, item[1]:item[2], item[0]] = 1
            gt_array[1, item[1]:item[2], item[0]] = 1
        for item in self.predict.numpy():
            pred_array[0, item[1]-2:item[1]+2, item[0]] = 1
            pred_array[1, item[1]-2:item[1]+2, item[0]] = 1
        import pdb; pdb.set_trace()
        er_func = er_score()
        er = er_func(gt_array, pred_array)
        er = tf.reduce_mean(er)
        assert er == 1.2

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

