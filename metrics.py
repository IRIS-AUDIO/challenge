import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import json
from tensorflow.keras.callbacks import *
from sj_train import cos_sim
from utils import *


class Custom_Metrics(Callback):
    def __init__(self, validation_data, loss_type):
        self.validation_data = validation_data
        self.F1_metric = tfa.metrics.F1Score(num_classes=3, threshold=0.5, average='micro')
        if loss_type == 'BCE':
            self.loss = tf.keras.losses.BinaryCrossentropy()
        if loss_type == 'FOCAL':
            self.loss = sigmoid_focal_crossentropy

    def on_epoch_end(self, epoch, logs):
        ER = tf.keras.metrics.Mean()
        F1 = tf.keras.metrics.Mean()
        COS_SIM = tf.keras.metrics.Mean()
        LOSS = tf.keras.metrics.Mean()
        for i, item in enumerate(self.validation_data):
            temp_targ = item[1]
            temp_pred = tf.convert_to_tensor(self.model.predict(item[0]))
            loss = self.loss(temp_targ, temp_pred)
            er = get_custom_er(temp_targ, temp_pred)
            f1 = self.F1_metric(temp_targ, temp_pred)
            cos_sim_score = cos_sim(temp_targ, temp_pred)
            f1_score = F1.update_state(f1)
            if er != 0:
                ER.update_state(er)
            COS_SIM.update_state(cos_sim_score)
            LOSS.update_state(loss)
            if i == 10:
                break
        logs['val_loss'] = LOSS.result().numpy()
        logs['val_er'] = ER.result().numpy()
        print(f"val_er: {ER.result().numpy()}, val_f1: {F1.result().numpy()}, COS_SIM: {COS_SIM.result().numpy()}, LOSS: {LOSS.result().numpy()}")
        return


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
        cls0, cls1, cls2 = self.get_start_end_frame(second_true)
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

def extract_middle(x):
    # [batch, time, cls]
    right_begin = tf.clip_by_value(
        x - tf.pad(x, [[0, 0], [0, 1], [0, 0]])[:, 1:], 0, 1)
    left_begin = tf.clip_by_value(
        x - tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1], 0, 1)

    starts = tf.where(left_begin)
    ends = tf.where(right_begin)
    starts = tf.gather(starts, tf.argsort(starts[:, -1]), -1)
    starts = tf.gather(starts, tf.argsort(starts[:, 0]), 0)
    ends = tf.gather(ends, tf.argsort(ends[:, -1]), -1)
    ends = tf.gather(ends, tf.argsort(ends[:, 0]), 0)

    middle = tf.cast((starts+ends)/2, tf.int32)
    result = tf.ones((tf.shape(middle)[0], tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.float32)
    result *= tf.one_hot(middle[:, 0], tf.shape(x)[0])[:, :, None, None]
    result *= tf.one_hot(middle[:, 1], tf.shape(x)[1])[:, None, :, None]
    result *= tf.one_hot(middle[:, 2], tf.shape(x)[2])[:, None, None, :]
    result = tf.reduce_max(result, axis=0)
    return result

def get_custom_er(gt, preds):
    metric = Challenge_Metric()
    total_score = 0 
    count = 0
    for gt_, preds_ in zip(gt, preds):
        ans0, ans1, ans2 = metric.get_start_end_time(gt_)
        if (ans0.shape[0] + ans1.shape[0] + ans2.shape[0]) != 0:
            smoothing_kernel_size = int(0.5 * 16000) // 256 # 0.5
            preds_ = tf.signal.frame(preds_, smoothing_kernel_size, 1, pad_end=True, axis=-2)
            preds_ = tf.reduce_mean(preds_, -2)
            preds_ = tf.cast(preds_ >= 0.5, tf.float32)
            cls0, cls1, cls2 = metric.get_start_end_time(preds_)

            cls0 = tf.expand_dims(tf.cast((cls0[...,-1] + cls0[...,-2])/2, tf.int32), -1)
            cls1 = tf.expand_dims(tf.cast((cls1[...,-1] + cls1[...,-2])/2, tf.int32), -1)
            cls2 = tf.expand_dims(tf.cast((cls2[...,-1] + cls2[...,-2])/2, tf.int32), -1)

            ans_0 = tf.tile(tf.expand_dims(ans0, 0), [cls0.shape[0], 1, 1]) # P, G, 2
            cls_0 = tf.tile(tf.expand_dims(cls0, 1), [1, ans0.shape[0], 1]) # P, G, 1
            cls0_ans = tf.cast((ans_0[:,:,0] <= cls_0[:,:,0]),tf.int32)*\
                tf.cast((cls_0[:,:,0] <= ans_0[:,:,1]),tf.int32)
            cls0_ans = tf.reduce_sum(tf.cast(tf.reduce_sum(cls0_ans, axis=-2) > 0, tf.int32), axis=-1)

            ans_1 = tf.tile(tf.expand_dims(ans1, 0), [cls1.shape[0], 1, 1]) # P, G, 2
            cls_1 = tf.tile(tf.expand_dims(cls1, 1), [1, ans1.shape[0], 1]) # P, G, 1
            cls1_ans = tf.cast((ans_1[:,:,0] <= cls_1[:,:,0]),tf.int32)*\
                tf.cast((cls_1[:,:,0] <= ans_1[:,:,1]),tf.int32)
            cls1_ans = tf.reduce_sum(tf.cast(tf.reduce_sum(cls1_ans, axis=-2) > 0, tf.int32), axis=-1)

            ans_2 = tf.tile(tf.expand_dims(ans2, 0), [cls2.shape[0], 1, 1]) # P, G, 2
            cls_2 = tf.tile(tf.expand_dims(cls2, 1), [1, ans2.shape[0], 1]) # P, G, 1
            cls2_ans = tf.cast((ans_2[:,:,0] <= cls_2[:,:,0]),tf.int32)*\
                tf.cast((cls_2[:,:,0] <= ans_2[:,:,1]),tf.int32)
            cls2_ans = tf.reduce_sum(tf.cast(tf.reduce_sum(cls2_ans, axis=-2) > 0, tf.int32), axis=-1)
            
            total_score += (cls0.shape[0] + cls1.shape[0] + cls2.shape[0] + (ans0.shape[0] + ans1.shape[0] + ans2.shape[0]) \
                - 2*(cls0_ans + cls1_ans + cls2_ans))/\
                    (ans0.shape[0] + ans1.shape[0] + ans2.shape[0])
            count += 1
    if count != 0:
        total_score /= count
    else:
        total_score = 0 
    return total_score

def get_custom_er_new(gt, preds):
    metric = Challenge_Metric()
    total_score = 0 
    smoothing_kernel_size = int(0.5 * 16000) // 256 # 0.5
    preds_ = tf.signal.frame(preds, smoothing_kernel_size, 1, pad_end=True, axis=-2)
    preds_ = tf.reduce_mean(preds_, -2)
    preds_ = tf.cast(preds_ >= 0.5, tf.float32)
    mids = extract_middle(preds_)
    correct = gt * mids 
    paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
    gt_temp = tf.pad(gt, paddings)[...,:-1,:]
    gt_ = tf.math.ceil(tf.reduce_sum(tf.cast(gt != gt_temp, dtype=tf.int32), axis=-2) / 2)
    ans_num = tf.cast(tf.reduce_sum(gt_), dtype=tf.float32) 
    preds_num = tf.reduce_sum(mids)
    correct_num = tf.reduce_sum(correct)
    total_score = (ans_num + preds_num - 2*correct_num) / ans_num
    return total_score

def get_er(gt, predict):
    predict_2 = tf.identity(predict)
    predict_2 = tf.gather(predict_2, tf.argsort(predict_2[:,1]))
    gt = tf.gather(gt, tf.argsort(gt[:,1]))
    N = len(predict_2) + len(gt)
    pred_N = len(predict_2)
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
    
def output_to_metric(cls0, cls1, cls2):
    answer_list = tf.cast(tf.zeros([0,2]), tf.int32)

    for item in cls0:
        new_item = tf.cast(tf.stack([0, (item[0] + item[1]) // 2], 0), item.dtype)[tf.newaxis, ...]
        answer_list = tf.concat([answer_list, new_item], axis=0)

    for item in cls1:
        new_item = tf.cast(tf.stack([1, (item[0] + item[1]) // 2], 0), item.dtype)[tf.newaxis, ...]
        answer_list = tf.concat([answer_list, new_item], axis=0)

    for item in cls2:
        new_item = tf.cast(tf.stack([2, (item[0] + item[1]) // 2], 0), item.dtype)[tf.newaxis, ...]
        answer_list = tf.concat([answer_list, new_item], axis=0)

    return answer_list
