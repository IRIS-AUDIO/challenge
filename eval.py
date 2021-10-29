from glob import glob
import tensorflow as tf

from data_utils import load_wav
from transforms import *
from utils import *

from sj_train import get_model, ARGS
from metrics import Challenge_Metric


def minmax_log_on_mel(mel, labels=None):
    # batch-wise pre-processing
    axis = tuple(range(1, len(mel.shape)))

    # MIN-MAX
    mel_max = tf.math.reduce_max(mel, axis=axis, keepdims=True)
    mel_min = tf.math.reduce_min(mel, axis=axis, keepdims=True)
    mel = safe_div(mel-mel_min, mel_max-mel_min)

    # LOG
    mel = tf.math.log(mel + EPSILON)

    if labels is not None:
        return mel, labels
    return mel


if __name__ == "__main__":
    config = ARGS().get()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    model = get_model(config)
    metric = Challenge_Metric()
    model.load_weights(f'{config.name}.h5')

    for path in sorted(glob('*.wav')):
        inputs = load_wav(path)
        inputs = complex_to_magphase(inputs)
        inputs = magphase_to_mel(config.n_mels)(inputs)
        inputs = minmax_log_on_mel(inputs)
        frame_len = inputs.shape[-2]
        overlap_hop = 512

        inputs = tf.signal.frame(inputs, config.n_frame, overlap_hop, pad_end=True, axis=-2)
        inputs = tf.transpose(inputs, (1, 0, 2, 3))
        preds = model.predict(inputs[..., :1]) # [batch, time, class]

        preds = tf.transpose(preds, [2, 0, 1])
        total_counts = tf.signal.overlap_and_add(tf.ones_like(preds), overlap_hop)[..., :frame_len]
        preds = tf.signal.overlap_and_add(preds, overlap_hop)[..., :frame_len]
        preds /= total_counts
        preds = tf.transpose(preds, [1, 0])

        # smoothing
        smoothing_kernel_size = int(0.5 * 16000) // 256 # 0.5초 길이의 kernel
        preds = tf.signal.frame(preds, smoothing_kernel_size, 1, pad_end=True, axis=-2)
        preds = tf.reduce_mean(preds, -2)

        preds = tf.cast(preds >= 0.5, tf.float32)
        cls0, cls1, cls2 = metric.get_start_end_time(preds)

        print()
        print(path)
        for i in cls0:
            time = tf.reduce_mean(tf.cast(i, tf.float32))
            print(f'class man: ({int(time//60)} : {int(time%60)})')
        for i in cls1:
            time = tf.reduce_mean(tf.cast(i, tf.float32))
            print(f'class woman: ({int(time//60)} : {int(time%60)})')
        for i in cls2:
            time = tf.reduce_mean(tf.cast(i, tf.float32))
            print(f'class kid: ({int(time//60)} : {int(time%60)})')
        metric.reset_state()

