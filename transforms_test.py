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

    def test_magphase_to_mel(self):
        # BATCH
        n_mels = 80
        magphase = np.random.randn(32, 257, 100, 4).astype('float32')
        mel = magphase_to_mel(n_mels)(magphase)
        self.assertEqual(mel.shape, [32, n_mels, 100, 2])

        # SINGLE SAMPLE
        magphase = np.random.randn(257, 100, 4).astype('float32')
        mel = magphase_to_mel(n_mels)(magphase)
        self.assertEqual(mel.shape, [n_mels, 100, 2])

    def test_log_magphase(self):
        specs = np.array([[  1,  10, 100,   0,   1,  -1],
                          [500,  50,   5,   3,  -3,   0]])
        t_specs = np.array([[0.      , 2.302585, 4.605170,  0,  1, -1],
                            [6.214608, 3.912023, 1.609438,  3, -3,  0]])
        self.assertAllClose(t_specs, log_magphase(specs, n_chan=3))

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

