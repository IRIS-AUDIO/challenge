from tensorflow.keras import Model
import efficientnet.model as efficientmodel
import pdb
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, BatchNormalization, GRU, Reshape, Input, TimeDistributed, Concatenate, GlobalAveragePooling1D

class model:
    def __init__(self, config):
        self.clf_model = getattr(efficientmodel, config.model)(
            include_top=False,
            input_tensor=Input(shape=(config.n_mels, None, 2)),
            weights=None,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )
        out = tf.transpose(self.clf_model.output, perm=[0, 2, 1, 3])
        out = Reshape([-1, out.shape[-1]*out.shape[-2]])(out)
        # out = Bidirectional(GRU(512))(out)
        out = GlobalAveragePooling1D()(out)
        if config.mode == 'clf':
            out1 = tf.expand_dims(Dense(256, activation='relu')(out), -2)
            out2 = tf.expand_dims(Dense(256, activation='relu')(out), -2)
            out3 = tf.expand_dims(Dense(256, activation='relu')(out), -2)

            out = Concatenate(axis=-2)([out1, out2, out3])
            out = Dense(10, activation='softmax')(out)
        elif config.mode == 'regr':
            out = Dense(64, activation='relu')(out)
            out = Dense(3, activation='sigmoid')(out)
            out *= 50
        self.clf_model = Model(inputs=self.clf_model.input, outputs=out)