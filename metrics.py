import json
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
import tensorflow_addons as tfa

from utils import *
from data_utils import *
from transforms import *


class eval_callback(tf.keras.callbacks.Callback):
    def __init__(self, config, NAME):
        super(eval_callback, self).__init__()
        self.config = config
        self.name = NAME
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 2:
            model = tf.keras.models.clone_model(self.model)
            model.load_weights(self.name)
            evaluate(self.config, model, verbose=True)
            

def evaluate(config, model, overlap_hop = 512, verbose: bool = False):
    final_score = []
    with open('sample_answer.json') as f:
        answer_gt = json.load(f)
    answer_gt = answer_gt['task2_answer']
    sr = 16000
    hop = 256
    metric = Challenge_Metric()

    for path in sorted(glob('*.wav')):
        inputs = load_wav(path)
        if config.n_chan == 3:
            inputs = stereo_mono(inputs)
        elif config.n_chan > 3:
            inputs = random_merge_aug(config.n_chan)(inputs, None)
        inputs = stft_filter(int(round(256 * 1000 / 16000)))(inputs)
        inputs = complex_to_magphase(inputs)
        inputs = magphase_to_mel(config.n_mels)(inputs)
        inputs = minmax(inputs)
        inputs = log_on_mel(inputs)
        frame_len = inputs.shape[-2]

        inputs = tf.signal.frame(inputs, config.n_frame, overlap_hop, pad_end=True, axis=-2)
        inputs = tf.transpose(inputs, (1, 0, 2, 3))
        preds = model.predict(inputs[..., :config.n_chan]) # [batch, time, class]
        
        if config.v in label_downsample_model:
            resolution = config.n_frame / preds.shape[-2]
            preds = tf.keras.layers.UpSampling1D(resolution)(preds)
            
        preds = tf.transpose(preds, [2, 0, 1])
        total_counts = tf.signal.overlap_and_add(tf.ones_like(preds), overlap_hop)[..., :frame_len]
        preds = tf.signal.overlap_and_add(preds, overlap_hop)[..., :frame_len]
        preds /= total_counts
        preds = tf.transpose(preds, [1, 0])

        # smoothing
        smoothing_kernel_size = int(0.5 * sr) // hop # 0.5초 길이의 kernel
        preds = tf.keras.layers.AveragePooling1D(smoothing_kernel_size, 1, padding='same')(preds[tf.newaxis, ...])[0]
        preds = tf.cast(preds >= 0.5, tf.float32)
        cls0, cls1, cls2 = metric.get_start_end_frame(preds)
        answer_gt_temp = tf.convert_to_tensor(answer_gt[os.path.basename(path)[:-4]])
        answer_predict = output_to_metric(hop, sr)(cls0, cls1, cls2)
        er = get_er(answer_gt_temp, answer_predict)
        
        final_score.append(er)
    if verbose:
        print('FINAL SCORE:', np.mean(final_score))
    return final_score

class Challenge_Metric:
    def __init__(self, sr=16000, hop=256) -> None:
        self.reset_state()
        self.sr = sr
        self.hop = hop

    def get_start_end_time(self, data):
        data1, data2, data3 = self.get_start_end_frame(data)
        data1 = tf.cast(tf.round(data1 * self.hop / self.sr), tf.int32)
        data2 = tf.cast(tf.round(data2 * self.hop / self.sr), tf.int32)
        data3 = tf.cast(tf.round(data3 * self.hop / self.sr), tf.int32)
        data1 = tf.gather(data1, np.unique(data1, True, axis=0)[1])
        data2 = tf.gather(data2, np.unique(data2, True, axis=0)[1])
        data3 = tf.gather(data3, np.unique(data3, True, axis=0)[1])
        return data1, data2, data3

    def get_start_end_frame(self, data):
        data_temp = tf.concat([tf.zeros([1,3]), data[:-1,:]], 0)
        diff_index = tf.where(data_temp != data)
        class_0 = diff_index[diff_index[:,1] == 0][:,0]
        class_1 = diff_index[diff_index[:,1] == 1][:,0]
        class_2 = diff_index[diff_index[:,1] == 2][:,0]

        if (class_0.shape[0] % 2 != 0):
            class_0 = tf.concat((class_0, tf.Variable([len(data)], dtype=tf.int64)),0)

        class_0 = tf.reshape(class_0, [-1, 2])
        class_0 = tf.transpose(tf.concat([[class_0[:,0]], [class_0[:,1] -1]], 0))

        if (class_1.shape[0] % 2 != 0):
            class_1 = tf.concat((class_1, tf.Variable([len(data)], dtype=tf.int64)),0)

        class_1 = tf.reshape(class_1, [-1, 2])
        class_1 = tf.transpose(tf.concat([[class_1[:,0]], [class_1[:,1] -1]], 0))

        if (class_2.shape[0]  % 2 != 0):
            class_2 = tf.concat((class_2, tf.Variable([len(data)], dtype=tf.int64)),0)

        class_2 = tf.reshape(class_2, [-1, 2])
        class_2 = tf.transpose(tf.concat([[class_2[:,0]], [class_2[:,1] -1]], 0))
        return class_0, class_1, class_2

    def get_second_answer(self, data):
        data_second = np.asarray([self.hop*i//self.sr for i in range(len(data))])
        second_true = np.zeros([np.max(data_second), 3])
        for i in range(np.max(data_second)):
            second_true[i, 0] = (tf.reduce_mean(data[:, 0][data_second == i]) > 0.5)
            second_true[i, 1] = (tf.reduce_mean(data[:, 1][data_second == i]) > 0.5)
            second_true[i, 2] = (tf.reduce_mean(data[:, 2][data_second == i]) > 0.5)
        cls0, cls1, cls2 = self.get_1(second_true)
        cls0 = tf.cast(cls0, dtype=tf.int32)
        cls1 = tf.cast(cls1, dtype=tf.int32)
        cls2 = tf.cast(cls2, dtype=tf.int32)
        return cls0, cls1, cls2

    def reset_state(self):
        self.arr0 = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=False)
        self.arr1 = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=False)
        self.arr2 = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=False)
        self.tmp0 = tf.TensorArray(tf.int64, size=2, dynamic_size=True, clear_after_read=True)
        self.tmp1 = tf.TensorArray(tf.int64, size=2, dynamic_size=True, clear_after_read=True)
        self.tmp2 = tf.TensorArray(tf.int64, size=2, dynamic_size=True, clear_after_read=True)
        self.ts0 = 0 # tmp size
        self.ts1 = 0 # tmp size
        self.ts2 = 0 # tmp size


def extract_middle(y_pred):
    # [batch, time, cls]
    pred_starts = tf.clip_by_value(y_pred - tf.pad(y_pred, [[0, 0], [1, 0], [0, 0]])[:, :-1], 0, 1)
    pred_ends = tf.clip_by_value(y_pred - tf.pad(y_pred, [[0, 0], [0, 1], [0, 0]])[:, 1:], 0, 1)
    n_pred = tf.reduce_sum(tf.cast(pred_starts, tf.float32), (1, 2))
    pred_starts = tf.where(pred_starts)
    pred_ends = tf.where(pred_ends)
    pred_starts = tf.gather(pred_starts, tf.argsort(pred_starts[:, -1]), -1)
    pred_starts = tf.gather(pred_starts, tf.argsort(pred_starts[:, 0]), 0)
    pred_ends = tf.gather(pred_ends, tf.argsort(pred_ends[:, -1]), -1)
    pred_ends = tf.gather(pred_ends, tf.argsort(pred_ends[:, 0]), 0)

    middle = tf.cast((pred_starts+pred_ends)/2, tf.int64)
    return middle


def get_er(gt, predict):
    predict_2 = tf.identity(predict)
    predict_2 = tf.gather(predict_2, tf.argsort(predict_2[:,1]))
    gt = tf.gather(gt, tf.argsort(gt[:,1]))
    N = len(predict_2) + len(gt)
    answer = 0
    for gt_item in gt:
        remove = False
        for i, pred_item in enumerate(predict_2):
            if (gt_item[1] <= pred_item[1]) and (pred_item[1] <= gt_item[2]):
                if gt_item[0] == pred_item[0]:
                    answer += 2 
                    temp = i
                    remove = True
                    break
        if remove:
            predict_2 = tf.concat((predict_2[:i,:], predict_2[i+1:, :]), axis=0)
    return (N - answer) / len(gt)

def output_to_metric(hop, sr):
    hop = hop
    sr = sr
    def output_to_metric_(cls0, cls1, cls2):
        answer_list = tf.cast(tf.zeros([0,2]), tf.int32)

        for item in cls0:
            new_item = tf.cast(tf.stack([0, ((item[0] + item[1]) / 2)*hop/sr], 0), answer_list.dtype)[tf.newaxis, ...]
            answer_list = tf.concat([answer_list, new_item], axis=0)

        for item in cls1:
            new_item = tf.cast(tf.stack([1, ((item[0] + item[1]) / 2)*hop/sr], 0), answer_list.dtype)[tf.newaxis, ...]
            answer_list = tf.concat([answer_list, new_item], axis=0)

        for item in cls2:
            new_item = tf.cast(tf.stack([2, ((item[0] + item[1]) / 2)*hop/sr], 0), answer_list.dtype)[tf.newaxis, ...]
            answer_list = tf.concat([answer_list, new_item], axis=0)
        return answer_list
    return output_to_metric_


def er_score(threshold=0.5, smoothing=True):
    threshold = tf.constant(threshold, tf.float32)

    def er(y_true, y_pred):
        y_true = tf.cast(y_true >= threshold, tf.int32)
        if smoothing:
            smoothing_kernel_size = int(0.5 * 16000) // 256 # 0.5
            y_pred = tf.keras.layers.AveragePooling1D(smoothing_kernel_size, padding='same')(y_pred)
        y_pred = tf.cast(y_pred >= threshold, tf.int32)

        # True values
        # [batch, time, cls]
        true_starts = tf.clip_by_value(
            y_true - tf.pad(y_true, [[0, 0], [1, 0], [0, 0]])[:, :-1], 0, 1)
        true_ends = tf.clip_by_value(
            y_true - tf.pad(y_true, [[0, 0], [0, 1], [0, 0]])[:, 1:], 0, 1)
        n_true = tf.reduce_sum(tf.cast(true_starts, tf.float32), (1, 2))

        true_starts = tf.where(true_starts)
        true_ends = tf.where(true_ends)
        true_starts = tf.gather(true_starts, tf.argsort(true_starts[:, -1]), -1)
        true_starts = tf.gather(true_starts, tf.argsort(true_starts[:, 0]), 0)
        true_ends = tf.gather(true_ends, tf.argsort(true_ends[:, -1]), -1)
        true_ends = tf.gather(true_ends, tf.argsort(true_ends[:, 0]), 0)

        # prediction values
        pred_starts = tf.clip_by_value(
            y_pred - tf.pad(y_pred, [[0, 0], [1, 0], [0, 0]])[:, :-1], 0, 1)
        pred_ends = tf.clip_by_value(
            y_pred - tf.pad(y_pred, [[0, 0], [0, 1], [0, 0]])[:, 1:], 0, 1)
        n_pred = tf.reduce_sum(tf.cast(pred_starts, tf.float32), (1, 2))

        pred_starts = tf.where(pred_starts)
        pred_ends = tf.where(pred_ends)
        pred_starts = tf.gather(pred_starts, tf.argsort(pred_starts[:, -1]), -1)
        pred_starts = tf.gather(pred_starts, tf.argsort(pred_starts[:, 0]), 0)
        pred_ends = tf.gather(pred_ends, tf.argsort(pred_ends[:, -1]), -1)
        pred_ends = tf.gather(pred_ends, tf.argsort(pred_ends[:, 0]), 0)

        middle = tf.cast((pred_starts+pred_ends)/2, tf.int64)

        # correct: correct batch and cls (true, pred)
        correct = (
            true_starts[:, ::2, None]==tf.transpose(middle, (1, 0))[None, ::2])
        correct = tf.reduce_min(tf.cast(correct, tf.float32), axis=1)

        mid_time = tf.transpose(middle[:, 1:2], (1, 0))
        correct *= tf.cast(true_starts[:, 1:2] <= mid_time, tf.float32)
        correct *= tf.cast(true_ends[:, 1:2] >= mid_time, tf.float32)
        correct = tf.reduce_max(tf.pad(correct, [[0, 0], [0, 1]]), -1)

        correct_per_sample = tf.reduce_sum(
            tf.one_hot(true_starts[:, 0], tf.shape(y_pred)[0])*correct[:, None],
            0)
        score = n_true + n_pred - 2 * correct_per_sample
        score /= tf.clip_by_value(n_true, 1, tf.reduce_max(n_true))
        return score
    return er


def cos_sim(y_true, y_pred):
    if isinstance(y_true, tuple):
        y_true = y_true[0]
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    mask = tf.cast(
        tf.reduce_sum(y_true, axis=-2) > 0., tf.float32) # [None, 3]
    mask = safe_div(mask, tf.reduce_sum(mask, axis=-1, keepdims=True))
    return tf.reduce_sum(
        tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-2) * mask, 
        axis=-1)


def f1_score():
    f1_score_fn = tfa.metrics.F1Score(num_classes=3, threshold=0.5, average='micro')
    def f1_score(y_true, y_pred):
        if isinstance(y_true, tuple):
            y_true = y_true[0]
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        return f1_score_fn(y_true, y_pred)
    return f1_score

