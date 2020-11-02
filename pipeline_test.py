import os
import numpy as np
import tensorflow as tf
from pipeline import *


class PipelineTest(tf.test.TestCase):
    def setUp(self):
        self.freq = 257
        self.chan = 4
        self.n_classes = 30

    def test_merge_complex_specs(self):
        n_frame = 10

        background = np.random.randn(self.freq, 8, self.chan).astype('float32')

        n_voices = 4
        voices = np.random.randn(n_voices, self.freq, n_frame, self.chan)
        voices = voices.astype('float32')
        mask = tf.sequence_mask(np.random.randint(1, n_frame, size=n_voices),
                                n_frame)
        mask = tf.reshape(mask, (n_voices, 1, n_frame, 1))
        voices *= tf.cast(mask, tf.float32)
        labels = np.random.randint(1, n_frame, size=n_voices)
        labels = np.eye(self.n_classes, dtype='float32')[labels]

        n_noises = 2
        noises = np.random.randn(n_noises, self.freq, n_frame, self.chan)
        noises = noises.astype('float32')
        mask = tf.sequence_mask(np.random.randint(1, n_frame, size=n_noises),
                                n_frame)
        mask = tf.reshape(mask, (n_noises, 1, n_frame, 1))
        noises *= tf.cast(mask, tf.float32)

        spec, l = merge_complex_specs(background, 
                                      (voices, labels), 
                                      noises,
                                      n_frame=n_frame, 
                                      n_classes=self.n_classes)
        self.assertEqual(spec.shape, [self.freq, n_frame, self.chan]) 
        self.assertEqual(l.shape, [n_voices, n_frame, self.n_classes]) 

    def test_to_frame_labels(self):
        n_voices = 10
        n_frame = 30

        x = None
        y = np.random.randn(n_voices, n_frame, self.n_classes).astype('float32')

        _, y_ = to_frame_labels(x, y)
        self.assertEqual(y_.shape, [n_frame, self.n_classes])

    def test_to_class_labels(self):
        n_voices = 10
        n_frame = 30

        x = None
        y = np.random.randn(n_voices, n_frame, self.n_classes).astype('float32')

        _, y_ = to_class_labels(x, y)
        self.assertEqual(y_.shape, [30])

        _, y_ = to_class_labels(x, y[np.newaxis, :])
        self.assertEqual(y_.shape, [1, 30])

    def test_make_pipeline(self):
        n_frame = 30

        backgrounds = [np.random.randn(self.freq, 
                                       np.random.randint(1, n_frame*2), 
                                       self.chan)
                       for _ in range(30)]
        voices = [np.random.randn(self.freq,
                                  np.random.randint(1, n_frame//2),
                                  self.chan)
                  for _ in range(40)]
        labels = np.random.randint(self.n_classes, size=(40,))
        labels = np.eye(self.n_classes, dtype='float32')[labels]

        noises = [np.random.randn(self.freq,
                                  np.random.randint(1, n_frame//2),
                                  self.chan)
                  for _ in range(50)]

        pipeline = make_pipeline(backgrounds, 
                                 voices, 
                                 labels, 
                                 noises,
                                 n_frame=n_frame, 
                                 max_voices=4,
                                 max_noises=4,
                                 n_classes=self.n_classes)

        for s, l in pipeline.take(3):
            self.assertEqual(s.shape, [self.freq, n_frame, self.chan])
            self.assertEqual(l.shape, [4, n_frame, self.n_classes])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

