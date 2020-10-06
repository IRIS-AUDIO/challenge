import os
import numpy as np
import tensorflow as tf
from models import *


class ModelsTest(tf.test.TestCase):
    def test_dot_product_attention(self):
        batch_size = 16
        seq_q, seq_kv = 30, 32
        dim_qk, dim_v = 20, 18
        q = np.random.randn(batch_size, seq_q, dim_qk).astype('float32')
        k = np.random.randn(batch_size, seq_kv, dim_qk).astype('float32')
        v = np.random.randn(batch_size, seq_kv, dim_v).astype('float32')

        out = dot_product_attention(q, k, v)
        self.assertAllEqual(out.shape, [batch_size, seq_q, dim_v])

    def test_multi_head_attention(self):
        batch_size = 16
        seq_q, seq_kv = 30, 32
        d_model = 20
        n_heads = 4

        q = np.random.randn(batch_size, seq_q, d_model).astype('float32')
        k = np.random.randn(batch_size, seq_kv, d_model).astype('float32')
        v = np.random.randn(batch_size, seq_kv, d_model).astype('float32')

        out = multi_head_attention(d_model, n_heads)(q, k, v)
        self.assertAllEqual(out.shape, [batch_size, seq_q, d_model])

        output_dim = 16
        out = multi_head_attention(d_model, n_heads, output_dim)(q, k, v)
        self.assertAllEqual(out.shape, [batch_size, seq_q, output_dim])

    def test_transformer_layer(self):
        batch_size = 16
        seq, d_model = 32, 20
        n_heads = 4

        x = np.random.randn(batch_size, seq, d_model).astype('float32')

        out = transformer_layer(d_model, n_heads, d_model*4)(x)
        self.assertAllEqual(out.shape, [batch_size, seq, d_model])

    def test_encoder(self):
        batch_size = 64
        seq = 90
        d_model = 128
        n_heads = 8
        output_dim = 16 # 15 + 1

        x = np.random.randn(batch_size, seq, d_model).astype('float32')
        y = np.random.randn(batch_size, 30, output_dim).astype('float32')
        y = tf.nn.softmax(y).numpy()

        inp = tf.keras.layers.Input((seq, d_model))
        out = encoder(2, d_model, n_heads)(inp)
        m = tf.keras.Model(inputs=inp, outputs=out)
        m.compile('adam', 'categorical_crossentropy')
        m.fit(x, y, epochs=5, verbose=0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

