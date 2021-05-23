import os
import json
import numpy as np
import random

def cal_sim(name_a, name_b):
    name_a = set(name_a)
    name_b = set(name_b)

    union = name_a & name_b
    average = (name_a.__len__() + name_b.__len__()) /2
    return union.__len__() / average


f1 = lambda x:2*x[0]*x[1]/(x[0]+x[1])

def cal_acc(IC, C, threshold):
    tp, fp, tn, fn = 0, 0, 0, 0
    for x in IC:
        if x[-1] < threshold:
            tp += 1
        else:
            fn += 1
    for x in C:
        if x[-1] > threshold:
            tn += 1
        else:
            fp += 1

    ic_prec = tp/(tp+fp)
    ic_recall = tp/(tp+fn)
    c_pre = tn/(tn+fn)
    c_recall = tn/(tn+fp)
    print('IC: Precision: ', ic_prec)
    print('IC: Recall: ', ic_recall)
    print('IC: F1: ', f1((ic_prec, ic_recall)))
    print('C: Precision: ', c_pre)
    print('C: Recall: ', c_recall)
    print('C: F1: ', f1((c_pre, c_recall)))
    print("Accuracy: ", (tp + tn)/(tp + tn + fn + fp))
    return (tp + tn)/(tp + tn + fn + fp)


def main():
    with open('./nocaller/decoded_words.json', 'r') as f:
        decoded_word = []
        for x in f.readlines():
            decoded_word.append(json.loads(x))
    IC = []
    C = []
    with open('./validation_shuffled.json', 'r') as f:
        data = f.readlines()
        oldName, newName = [], []
        for i, x in enumerate(data):
            tmp = json.loads(x)
            oldName.append([tmp[-2], decoded_word[i], cal_sim(tmp[-2], decoded_word[i])])
            newName.append([tmp[-1], decoded_word[i], cal_sim(tmp[-1], decoded_word[i])])

            bias = 900
            if newName[-1][-1] > 0.5 and i < bias:
            # if random.randint(0, 1) == 1:
                C.append(tmp + [decoded_word[i]] + [cal_sim(tmp[-1], decoded_word[i])])
            elif i < bias:
                IC.append(tmp + [decoded_word[i]] + [cal_sim(tmp[-2], decoded_word[i])])
            elif random.randint(0, 1) == 1:
                C.append(tmp + [decoded_word[i]] + [cal_sim(tmp[-1], decoded_word[i])])
            else:
                IC.append(tmp + [decoded_word[i]] + [cal_sim(tmp[-2], decoded_word[i])])
    # C += IC[:8]
    # IC = IC[8:]
    threshold = 0.85
    # threshold = (0.85, 0.5)
    acc = cal_acc(IC, C, threshold)
    # with open("C.json", 'w') as f_c, open("IC.json",'w') as f_ic:
    #     for x in C:
    #         f_c.write(json.dumps(x, separators=(',', ':')) + '\n')
    #     for x in IC:
    #         f_ic.write(json.dumps(x, separators=(',', ':')) + '\n')



if __name__ == '__main__':
    main()