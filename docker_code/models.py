# https://www.tensorflow.org/tutorials/text/transformer?hl=en
import tensorflow as tf
import tensorflow.keras.backend as K


def dot_product_attention(q, k, v):
    '''
    q: (..., seq_len_q, depth)
    k: (..., seq_len_kv, depth)
    v: (..., seq_len_kv, depth_v)
    '''
    logits = tf.matmul(q, k, transpose_b=True)
    logits /= tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))

    weights = tf.nn.softmax(logits)
    output = tf.matmul(weights, v) 

    return output


def multi_head_attention(d_model, n_heads, output_dim=None, perm_and_reshape=True):
    assert d_model % n_heads == 0
    depth = d_model // n_heads
    if output_dim is None:
        output_dim = d_model

    def mha(q, k, v):
        q = tf.keras.layers.Dense(d_model)(q) # (batch_size, seq_q, d_model)
        k = tf.keras.layers.Dense(d_model)(k) # (batch_size, seq_kv, d_model)
        v = tf.keras.layers.Dense(d_model)(v) # (batch_size, seq_kv, d_model)

        # split heads (batch, seq, d_model) -> (batch, n_heads, seq, depth)
        q = tf.keras.layers.Reshape((-1, n_heads, depth))(q)
        q = tf.keras.layers.Permute((2, 1, 3))(q)
        k = tf.keras.layers.Reshape((-1, n_heads, depth))(k)
        k = tf.keras.layers.Permute((2, 1, 3))(k)
        v = tf.keras.layers.Reshape((-1, n_heads, depth))(v)
        v = tf.keras.layers.Permute((2, 1, 3))(v)

        attn = dot_product_attention(q, k, v)

        if perm_and_reshape:
            attn = tf.keras.layers.Permute((2, 1, 3))(attn)
            attn = tf.keras.layers.Reshape((-1, d_model))(attn)
            attn = tf.keras.layers.Dense(output_dim)(attn)

        return attn
    return mha


def transformer_layer(d_model, n_heads, dff=None, rate=0.1):
    if dff is None:
        dff = d_model * 4

    def _transformer_layer(x):
        attn = multi_head_attention(d_model, n_heads)(x, x, x)
        attn = tf.keras.layers.Dropout(rate)(attn)
        x = tf.keras.layers.LayerNormalization()(x + attn)

        # FFN
        ffn = tf.keras.layers.Dense(dff, activation='relu')(x)
        ffn = tf.keras.layers.Dense(d_model)(ffn)
        ffn = tf.keras.layers.Dropout(rate)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)

        return x
    return _transformer_layer


def encoder(n_layers, d_model, n_heads, dff=None, rate=0.1, softmax=False):
    if dff is None:
        dff = d_model * 4

    def _encoder(x):
        # x = tf.keras.layers.Dropout(rate)(x)
        
        # Two Embeddings (3 for classes, 10 for degrees)
        cls = K.expand_dims(K.arange(3), axis=0)
        cls = K.stop_gradient(cls)
        cls = tf.keras.layers.Embedding(3, d_model)(cls)
        cls = K.expand_dims(cls, axis=2) # (1, 3, 1, d_model)

        direct = K.expand_dims(K.arange(10), axis=0)
        direct = K.stop_gradient(direct)
        direct = tf.keras.layers.Embedding(10, d_model)(direct)
        direct = K.expand_dims(direct, axis=1) # (1, 1, 10, d_model)

        embedding = tf.keras.layers.Reshape((30, d_model))(cls + direct)

        for i in range(n_layers):
            x = transformer_layer(d_model, n_heads, dff, rate)(x)

        x = multi_head_attention(d_model, n_heads, perm_and_reshape=False)(embedding, x, x)
        x = tf.keras.layers.Dropout(rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if softmax:
            x = tf.keras.layers.Softmax()(x)

        return x
    return _encoder
