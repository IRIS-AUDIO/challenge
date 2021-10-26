import numpy as np
import tensorflow as tf
import json


# label for metric (Class, Start, End)
# predict(Class, Time)
def get_er(predict, gt):
    predict.sort(key = lambda x:x[1])
    gt.sort(key = lambda x:x[1])
    N = len(predict)
    S = 0
    D = 0
    I = 0
    reverse_I = len(predict)
    for gt_item in gt:
        temp_list = []
        for pred_item in predict: # get all item between pred time
            if (gt_item[1] <= pred_item[1]) and (pred_item[1] <= gt_item[2]):
                temp_list.append(pred_item)
                reverse_I -= 1
        if len(temp_list) == 0:
            D += 1
        elif temp_list[0][0] != gt_item[0]: # check first item if wrong count S
            S += 1
            I += len(temp_list)-1 # else item are all 
        else:
            I += len(temp_list)-1 # else item are all 
    I += reverse_I # case for wrong insertion 
    return (S + D + I) / N

if __name__ == '__main__':
    gt = [[0, 10, 20],  [2, 45, 64], [1, 25, 32]]
    predict = [[1,15], [1,16], [2, 22], [2,47], [1, 34], [1,44], [2, 65]]
    print(get_er(predict, gt) == 1) # S=1, D =2, I=4, N=7
