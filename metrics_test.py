import os
import numpy as np
import tensorflow as tf
from metrics import *


class MetricsTest(tf.test.TestCase):
    def setUp(self):
        self.gt = [[0, 0, 10],  [2, 0, 20], [1, 15, 30], [2, 31, 40],  [1, 27, 32]]
        self.predict = [[1,5], [1, 19], [2, 32], [2,38], [0, 38]]

    def test_get_er(self):
        self.assertEqual(1.2, get_er(self.predict, self.gt))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

