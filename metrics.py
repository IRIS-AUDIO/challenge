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
    