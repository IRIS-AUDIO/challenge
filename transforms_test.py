import os
import torch
import torchaudio
import numpy as np
import tensorflow as tf
from transforms import *


class TransformsTest(tf.test.TestCase):
    def test_mask(self):
        tf.random.set_seed(100)
        org = np.array([[ 0,  1,  2,  3,  4],
                        [ 5,  6,  7,  8,  9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24]])
        target = np.array([[ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  0,  0],
                           [15, 16, 17, 18, 19],
                           [20, 21, 22, 23, 24]])
        self.assertAllEqual(target, 
                            mask(org, axis=0, max_mask_size=None, n_mask=1))

        tf.random.set_seed(2020)
        target = np.array([[ 0,  1,  0,  3,  4],
                           [ 0,  6,  0,  8,  9],
                           [ 0, 11,  0, 13, 14],
                           [ 0, 16,  0, 18, 19],
                           [ 0, 21,  0, 23, 24]])
        self.assertAllEqual(target, 
                            mask(org, axis=1, max_mask_size=3, n_mask=2))

    def test_random_shift(self):
        tf.random.set_seed(0)
        org = np.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])
        target = np.array([[3, 4, 5],
                           [6, 7, 8],
                           [0, 0, 0]])
        self.assertAllEqual(target, 
                            random_shift(org, axis=0, width=2))

    def test_random_magphase_flip(self):
        tf.random.set_seed(0)
        # INPUTS
        spec = np.array([[[ 1,  2, -1, -2],
                          [ 3,  4, -3, -4],
                          [ 5,  6,  0,  4]],
                         [[ 1,  2,  3,  4],
                          [ 3,  6,  9, 12],
                          [10, 11, 12, 13]]])
        label = np.array([[0, 0, 1],
                          [1, 0, 0]])
        # TARGETS
        t_spec = np.array([[[ 2,  1, -2, -1],
                            [ 4,  3, -4, -3],
                            [ 6,  5,  4,  0]],
                           [[ 2,  1,  4,  3],
                            [ 6,  3, 12,  9],
                            [11, 10, 13, 12]]])
        t_label = np.array([[0, 0, 1],
                            [0, 1, 0]])
        
        s, l = random_magphase_flip(spec, label)
        self.assertAllEqual(t_spec, s)
        self.assertAllEqual(t_label, l)

    def test_magphase_mixup(self):
        tf.random.set_seed(0)
        specs = np.array([[[  1., 10.,  0.,  0.],
                           [ 10.,  5., -1., -3.]],
                          [[  5.,  5.,  3., -3.],
                           [ 10.,100.,  0.,  1.]]])
        labels = np.array([[0., 1., 0.],
                           [1., 0., 0.]])

        t_specs = np.array([[[ 1.788283, 3.087293, 2.957654,-0.106144],
                             [ 8.782782,44.648120,-0.539802, 1.045481]],
                            [[ 2.224290, 2.015974, 2.970586,-0.188880],
                             [ 8.782782,52.159332,-0.460198, 1.033636]]])
        t_labels = np.array([[0.463552, 0.536448, 0.      ],
                             [0.536448, 0.463552, 0.      ]])

        s, l = magphase_mixup(alpha=2.)(specs, labels)
        self.assertAllClose(t_specs, s)
        self.assertAllClose(t_labels, l)

        # feat == 'complex'
        tf.random.set_seed(0)
        specs = magphase_to_complex(specs)
        s, l = magphase_mixup(alpha=2., feat='complex')(specs, labels)
        self.assertAllClose(t_specs, s)
        self.assertAllClose(t_labels, l)

    def test_log_magphase(self):
        specs = np.array([[  1,  10, 100,   0,   1,  -1],
                          [500,  50,   5,   3,  -3,   0]])
        t_specs = np.array([[0.      , 2.302585, 4.605170,  0,  1, -1],
                            [6.214608, 3.912023, 1.609438,  3, -3,  0]])
        self.assertAllClose(t_specs, log_magphase(specs))

    def test_minmax_norm_magphse(self):
        n_sample, n_feature, n_chan = 5, 10, 2
        axis = tuple(range(1, 3))
        mag = np.random.randn(n_sample, n_feature, n_chan)
        phase = np.random.rand(n_sample, n_feature, n_chan)
        phase = (2*phase - 1) * np.pi
        magphase = np.concatenate([mag, phase], axis=-1)

        minmax_normed = minmax_norm_magphase(magphase)
        mins = tf.math.reduce_min(minmax_normed, axis=axis)
        maxs = tf.math.reduce_max(minmax_normed, axis=axis)

        self.assertAllClose(mins, tf.zeros_like(mins))
        self.assertAllClose(maxs, tf.ones_like(maxs))

    def test_complex_to_magphase(self):
        complex_tensor = np.array(
            [[1, 0], [0, 1], [-1, 0], [0, -1]], dtype='float32')
        magphase = np.array(
            [[1, 0], [1, np.pi/2], [1, np.pi], [1, -np.pi/2]],
            dtype='float32')

        self.assertAllClose(magphase,
                            complex_to_magphase(complex_tensor))

    def test_magphase_to_complex(self):
        magphase = np.array(
            [[1, 0], [1, np.pi/2], [1, np.pi], [1, -np.pi/2]],
            dtype='float32')
        complex_tensor = np.array(
            [[1, 0], [0, 1], [-1, 0], [0, -1]], dtype='float32')

        self.assertAllClose(complex_tensor, magphase_to_complex(magphase))

    def test_phase_vocoder(self):
        n_freq, time, chan2 = 257, 100, 6
        complex_spec = tf.random.normal([n_freq, time, chan2])

        self.assertAllEqual(complex_spec,
                            phase_vocoder(complex_spec, 1.))
        
        for rate in [1.2, 0.8]:
            pv = phase_vocoder(complex_spec, rate=rate)
            self.assertAllEqual([n_freq, int(np.ceil(time/rate)), chan2],
                                pv.shape)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()
