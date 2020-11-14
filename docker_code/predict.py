from params import getArgs
from data_utils import load_wav
import pdb, json


def tojson(data):
    with open('t3_res_jiho.chang@kriss.re.kr.json', 'w') as f:
        json.dump(data, f, indent='\t')

def main(config):
    wavs = load_wav(config)

    import tensorflow as tf
    import efficientnet.model as model
    x = tf.keras.layers.Input(shape=(config.n_mels, None, 2))
    model = getattr(model, config.model)(
        include_top=False,
        weights=None,
        input_tensor=x,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils,
    )
    out = tf.transpose(model.output, perm=[0, 2, 1, 3])
    out = tf.keras.layers.Reshape([-1, out.shape[-1]*out.shape[-2]])(out)

    if config.n_layers > 0:
        if config.mode == 'GRU':
            out = tf.keras.layers.Dense(config.n_dim)(out)
            for i in range(config.n_layers):
                # out = transformer_layer(config.n_dim, config.n_heads)(out)
                out = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(config.n_dim, return_sequences=True),
                    backward_layer=tf.keras.layers.GRU(config.n_dim, 
                                                       return_sequences=True,
                                                       go_backwards=True))(out)
        elif config.mode == 'transformer':
            out = tf.keras.layers.Dense(config.n_dim)(out)
            out = encoder(config.n_layers,
                          config.n_dim,
                          config.n_heads)(out)

            out = tf.keras.layers.Flatten()(out)
            out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Dense(config.n_classes, activation='relu')(out)
    model = tf.keras.models.Model(inputs=model.input, outputs=out)

    model.load_weights('model.h5')

    wavs = model.predict(wavs)
    wavs = wavs / config.multiplier
    wavs = tf.reshape(wavs, [*wavs.shape[:2], 3, 10])

    angles = tf.cast(tf.round(tf.reduce_sum(wavs, axis=(1, 2))), tf.int8)
    classes = tf.cast(tf.round(tf.reduce_sum(wavs, axis=(1, 3))), tf.int8)

    data = {
        'track3_results':list()
    }
    for idx, (ag, cl) in enumerate(zip(angles, classes)):
        _data = {'id':idx,'angle':ag.numpy().tolist(), 'class':cl.numpy().tolist()}
        data['track3_results'].append(_data)
    tojson(data)

if __name__ == "__main__":
    import sys
    main(getArgs(sys.argv[1:]))