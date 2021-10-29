import numpy as np
import tensorflow as tf
import json


# label for metric (Class, Start, End)
# predict(Class, Time)
def get_er(gt, predict):
    predict.sort(key = lambda x:x[1])
    gt.sort(key = lambda x:x[1])
    N = len(predict) + len(gt)
    pred_N = len(predict)
    answer = 0
    for gt_item in gt:
        remove = False
        for i, pred_item in enumerate(predict):
            if (gt_item[1] <= pred_item[1]) and (pred_item[1] <= gt_item[2]):
                if gt_item[0] == pred_item[0]:
                    answer += 2 
                    temp = pred_item
                    remove = True
                    break
        if remove:
            predict.remove(temp)
    return (N - answer) / pred_N
    

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
        # data (frame, class)
        idx = tf.where(data != 0)
        for frame in tf.range(idx.shape[0], dtype=tf.int32):
            class_num = idx[frame][-1]
            frame_num = idx[frame][0]
            arr = getattr(self, f'arr{class_num}')
            tmp = getattr(self, f'tmp{class_num}')
            ts = getattr(self, f'ts{class_num}')
            if ts == 1:
                st = tmp.read(0)
                ts -= 1
                if frame_num - st == 1:
                    tmp.write(0, st).mark_used()
                    tmp.write(1, frame_num).mark_used()
                    ts += 2
                elif frame_num - st > 1:
                    arr.write(arr.size(), tf.stack([st, st])).mark_used()
                    tmp.write(0, frame_num).mark_used()
                    ts += 1
            elif ts == 2:
                st = tmp.read(0)
                end = tmp.read(1)
                ts -= 2
                if frame_num - end == 1:
                    tmp.write(0, st).mark_used()
                    tmp.write(1, frame_num).mark_used()
                    ts += 2
                elif frame_num - end > 1:
                    arr.write(arr.size(), tf.stack([st, end])).mark_used()
                    tmp.write(0, frame_num).mark_used()
                    ts += 1
            elif ts == 0:
                tmp.write(0, frame_num).mark_used()
                ts += 1
            setattr(self, f'ts{class_num}', ts)

        for i in tf.range(3):
            arr = getattr(self, f'arr{i}')
            tmp = getattr(self, f'tmp{i}')
            ts = getattr(self, f'ts{i}')
            if ts == 1:
                arr.write(arr.size(), [tmp.gather(tf.range(ts))[0], tmp.gather(tf.range(ts))[0]]).mark_used()
            elif ts == 2:
                arr.write(arr.size(), tmp.gather(tf.range(ts))).mark_used()
        return self.arr0.stack(), self.arr1.stack(), self.arr2.stack()

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

