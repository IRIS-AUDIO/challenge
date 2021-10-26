import numpy as np
import tensorflow as tf
import json


# label for metric (Class, Start, End)
# predict(Class, Time)
def get_er(predict, gt):
    predict.sort(key = lambda x:x[1])
    gt.sort(key = lambda x:x[1])
    N = len(predict) + len(gt)
    pred_N = len(predict)
    answer = 0
    for gt_item in gt:
        for i, pred_item in enumerate(predict):
            if (gt_item[1] <= pred_item[1]) and (pred_item[1] <= gt_item[2]):
                if gt_item[0] == pred_item[0]:
                    answer += 2 
                    predict.remove(pred_item)
                    break
    return (N - answer) / pred_N


if __name__ == '__main__':
    gt = [[0, 0, 10],  [2, 0, 20], [1, 15, 30], [2, 31, 40],  [1, 27, 32]]
    predict = [[1,5], [1, 19], [2, 32], [2,38], [0, 38]]
    print(get_er(predict, gt) == 1.2)
    
