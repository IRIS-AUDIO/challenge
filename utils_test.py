import numpy as np
import unittest
from utils import *


class UtilsTest(unittest.TestCase):
    def test_seq_to_windows(self):
        seq = np.array([1, 2, 3, 4, 5])
        window = np.array([-3, -1, 0, 1, 3])

        target = np.array([[0, 0, 1, 2, 4],
                           [0, 1, 2, 3, 5],
                           [0, 2, 3, 4, 0],
                           [1, 3, 4, 5, 0],
                           [2, 4, 5, 0, 0]])
        self.assertEqual(target.tolist(), 
                         seq_to_windows(seq, window).tolist())
        self.assertEqual(target[::2].tolist(),
                         seq_to_windows(seq, window, 2).tolist())

    def test_windows_to_seq(self):
        windows = np.array([[0, 0, 1, 2, 4],
                            [0, 1, 2, 3, 5],
                            [0, 2, 3, 4, 0],
                            [1, 3, 4, 5, 0],
                            [2, 4, 5, 0, 0]])
        window = np.array([-3, -1, 0, 1, 3])

        target = np.array([1, 2, 3, 4, 5])
        self.assertTrue(
            np.allclose(target, windows_to_seq(windows, window)))
        self.assertTrue(
            np.allclose(target, windows_to_seq(windows[::2], window, skip=2)))

    def test_list_to_generator(self):
        n_samples = 4
        x = np.random.randn(n_samples, 30)
        y = np.random.randn(n_samples)

        x_gen = list_to_generator(x)
        self.assertTrue(callable(x_gen))
        for i, x_ in enumerate(x_gen()):
            self.assertEqual(x[i].tolist(), x_.tolist())

        xy_gen = list_to_generator((x, y))
        self.assertTrue(callable(xy_gen))
        for i, (x_, y_) in enumerate(xy_gen()):
            self.assertEqual(x[i].tolist(), x_.tolist())
            self.assertEqual(y[i], y_)

    def test_load_data(self):
        raise NotImplemented('TODO: not yet implemented')

    def test_apply_kernel_regularizer(self):
        n_samples, in_shape, out_shape = 128, 4, 4
        x = np.random.randn(n_samples, in_shape)
        y = np.random.randint(out_shape, size=n_samples)

        # model without regularizer
        tf.random.set_seed(0)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(in_shape,)))
        model.add(tf.keras.layers.Dense(out_shape, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        model.fit(x, y, verbose=False)
        base_weights = model.weights[:]

        # model with regularizer
        tf.random.set_seed(0)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(in_shape,)))
        model.add(tf.keras.layers.Dense(out_shape, activation='softmax'))

        model = apply_kernel_regularizer(model, tf.keras.regularizers.l2(0.1))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        model.fit(x, y, verbose=False)
        new_weights = model.weights[:]

        for b, n in zip(base_weights, new_weights):
            self.assertNotEqual(b.numpy().tolist(), n.numpy().tolist())

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()
