import os
import numpy as np
import tensorflow as tf
from metrics import *


class MetricsTest(tf.test.TestCase):
    def test_D_direction(self):
        y_true = np.array([[0, 2, 0, 0, 1, 0, 1, 0, 0, 1],
                           [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]],
                          dtype='float32')
        y_pred = np.array([[1, 1, 0, 0, 2, 0, 0, 0, 1, 1],
                           [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]],
                          dtype='float32')

        target = np.array([1.9375, 0.], dtype='float32')
        self.assertAllClose(target, D_direction(y_true, y_pred))

    def test_D_class(self):
        y_true = np.array([[1, 2, 3],
                           [1, 0, 2]],
                          dtype='float32')
        y_pred = np.array([[1, 2, 3],
                           [2, 2, 2]],
                          dtype='float32')

        target = np.array([0., 5.], dtype='float32')
        self.assertAllClose(target, D_class(y_true, y_pred))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()
