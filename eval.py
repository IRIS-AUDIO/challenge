from glob import glob
import tensorflow as tf
import json

from transforms import *
from utils import *
from data_utils import *

from sj_train import get_model, ARGS, random_merge_aug, stereo_mono, stft_filter, label_downsample_model
from metrics import Challenge_Metric, output_to_metric, get_er, evaluate
    

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


def second2frame(seconds: list, frame_num, resolution):
    # seconds = [[class, start, end], ...]
    frames = np.zeros([frame_num, 3], dtype=np.float32)
    for second in seconds:
        class_num = second[0]
        start = int(np.round(second[1] * resolution))
        end = int(np.round(second[2] * resolution))
        frames[start:end,class_num] += 1
    return tf.convert_to_tensor(frames, dtype=tf.float32)



if __name__ == "__main__":
    config = ARGS()
    config.args.add_argument('--verbose', help='verbose', type=bool, default=True)
    config.args.add_argument('--p', help='parsing name', action='store_true')
    config.args.add_argument('--path', type=str, default='')
    config = config.get()
    if config.p:
        parsed_name = config.name.split('_')
        if parsed_name[0][0] not in ('B', 'v'):
            parsed_name = parsed_name[1:]
        if parsed_name[0] == 'vad':
            config.model_type = 'vad'
            config.model = 1
        else:
            config.model = int(parsed_name[0][-1])
        config.v = int(parsed_name[1][-1])
        config.n_mels = int(parsed_name[6][3:])
        config.n_chan = int(parsed_name[7][-1])
        config.n_frame = int(parsed_name[9].split('framelen')[-1])
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    model = get_model(config)
    model.load_weights(os.path.join(config.path, f'{config.name}.h5'))
    final_score = evaluate(config, model, verbose=config.verbose)

