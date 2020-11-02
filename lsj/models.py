from tensorflow.keras import Model
import efficientnet.model as efficientmodel
import pdb
import tensorflow as tf

class model:
    def __init__(self, config):
        self.clf_model = getattr(efficientmodel, config.model)(
            include_top=False,
            input_tensor=tf.keras.layers.Input(shape=(config.n_mels, None, 2)),
            weights=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )
        out = tf.transpose(self.clf_model.output, perm=[0, 2, 1, 3])
        out = tf.keras.layers.Reshape([-1, out.shape[-1]*out.shape[-2]])(out)
        out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(3))(out)
        out = tf.keras.layers.Dense(3,activation='sigmoid',use_bias=False)(out)

        self.clf_model = Model(inputs=self.clf_model.input, outputs=out)