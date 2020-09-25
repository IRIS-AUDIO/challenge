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

    def test_cutmix(self):
        xs = np.array([[[[0], [0], [0]],
                        [[0], [0], [0]],
                        [[0], [0], [0]]],
                       [[[1], [1], [1]],
                        [[1], [1], [1]],
                        [[1], [1], [1]]],
                       [[[2], [2], [2]],
                        [[2], [2], [2]],
                        [[2], [2], [2]]]])
        ys = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
        t_xs = np.array([[[[0], [1], [0]],
                          [[0], [0], [0]],
                          [[0], [0], [0]]],
                         [[[1], [2], [1]],
                          [[1], [1], [1]],
                          [[1], [1], [1]]],
                         [[[2], [0], [2]],
                          [[2], [2], [2]],
                          [[2], [2], [2]]]])
        t_ys = np.array([[8/9, 1/9, 0],
                         [0, 8/9, 1/9],
                         [1/9, 0, 8/9]])

        tf.random.set_seed(111)
        cutmix_xs, cutmix_ys = cutmix(xs, ys)
        self.assertAllClose(t_xs, cutmix_xs)
        self.assertAllClose(t_ys, cutmix_ys)

    def test_interclass(self):
        # TODO
        self.assertEqual(True, False)
        '''
        shape = (32, 16, 16, 3) # batch, height, width, chan
        n_classes = 4
        xs = np.random.randn(*shape)
        ys = np.random.randint(n_classes, size=shape[0])
        ys = np.eye(n_classes)[ys]

        inter_xs, inter_ys = interclass_cutmix(xs, ys)
        self.assertNotAllClose(xs, inter_xs)
        self.assertAllEqual(tf.reduce_max(inter_ys, axis=-1),
                            tf.ones(shape[0]))
        '''

    def test_interbinary(self):
        # TODO
        self.assertEqual(True, False)
        '''
        N_CLASSES = 5
        N_SAMPLES = 128
        tf.random.set_seed(0)
        specs = np.random.randn(N_SAMPLES, 16, 16)
        labels = np.random.randint(N_CLASSES, size=N_SAMPLES)
        labels = np.eye(N_CLASSES)[labels]

        s, l = bin_mixup(alpha=1.)(specs, labels)
        self.assertEqual(2, len(np.unique(l[..., -1]))) # only 0, 1
        self.assertEqual(4, len(np.unique(l[..., :-1]))) # + 2
        '''

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

    def test_merge_complex_specs(self):
        n_frame = 5
        n_voice_classes = 5

        background = np.array([[[1, 0], [-1, 0], [1, 0]],
                               [[-1, 0], [1, 0], [-1, 0]]],
                              dtype='float32')
        voice = np.array([[[0, 1], [0, -1]],
                          [[0, -1], [0, 1]]],
                         dtype='float32')

        tf.random.set_seed(0)
        s_target = np.array([[[ 1,  0.        ],
                              [-1,  0.        ],
                              [ 1,  0.40697938],
                              [ 1, -0.40697938],
                              [-1,  0.        ]],
                             [[-1,  0.        ],
                              [ 1,  0.        ],
                              [-1, -0.40697938],
                              [-1,  0.40697938],
                              [ 1,  0.        ]]])
        label = 3
        spec, l = merge_complex_specs(background, 
                                      (voice, label), 
                                      n_frame=n_frame, 
                                      n_voice_classes=n_voice_classes)
        self.assertAllClose(s_target, spec)
        self.assertAllClose([0, 0, 0, 1, 0, 0], l)

        # NON-VOICE AUDIO
        spec, l = merge_complex_specs(background, 
                                      (voice, label), 
                                      n_frame=3, 
                                      prob=0.,
                                      n_voice_classes=n_voice_classes)
        self.assertAllClose(background, spec)
        self.assertAllClose([0, 0, 0, 0, 0, 1], l)

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
