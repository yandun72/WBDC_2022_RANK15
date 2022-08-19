import numpy as np
double_0 = np.load('./double_device_0.npy') #双流没有预训练
double_1 = np.load('./double_device_1.npy')

double_2_0 = np.load('./double_2_device_0.npy')#双流有预训练
double_2_1 = np.load('./double_2_device_1.npy')

import json 
single_1_0 = np.load('./single_1_device_0.npy') #单流有预训练 qq浏览器
single_1_1 = np.load('./single_1_device_1.npy')

single_2_0 = np.load('./single_2_device_0.npy') #单流有预训练 albef式
single_2_1 = np.load('./single_2_device_1.npy')

# def sofmax(logits):
#     e_x = np.exp(logits)
#     probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
#     return probs
# double_0_softmax = sofmax(double_0)
# double_1_softmax = sofmax(double_0)

# single_2_0_softmax = sofmax(single_2_0)
# single_2_1_softmax = sofmax(single_2_1)

# single_1_0_softmax = sofmax(single_1_0)
# single_1_1_softmax = sofmax(single_1_1)

'''
单独测一下每一个模型
'''
#0.7154 双流没有预训练
# first_half = double_0
# final_half = double_1

#0.7204 双流有预训练
# first_half = double_2_0
# final_half = double_2_1


#0.720 单流有预训练 qq浏览器
# first_half = single_1_0
# final_half = single_1_1

#0.724 单流有预训练 albef式
# first_half = single_2_0
# final_half = single_2_1



'''
直接相加
'''
#0.72860
# first_half = double_0 + single_1_0
# final_half = double_1 + single_1_1

#0.7307
# first_half = double_0 + single_2_0
# final_half = double_1 + single_2_1

#731
# first_half = single_1_0 + single_2_0
# final_half = single_1_1 + single_2_1

#730
# first_half = double_0 + single_1_0 + single_2_0
# final_half = double_1 + single_1_1 + single_2_1

#0.7347
# first_half = double_2_0 + single_1_0 + single_2_0
# final_half = double_2_1 + single_1_1 + single_2_1


#0.7355 四个简单相加
# first_half = double_0 + double_2_0 +  single_1_0 + single_2_0
# final_half = double_1 + double_2_1 +  single_1_1 + single_2_1


'''
加权相加 0.731
'''
# first_half = 0.5*double_0 + 0.2*single_1_0 + 0.3*single_2_0
# final_half = 0.5*double_1 + 0.2*single_1_1 + 0.3*single_2_1

'''
加权相加 0.7327
'''
# first_half = 0.4*double_0 + 0.2*single_1_0 + 0.4*single_2_0
# final_half = 0.4*double_1 + 0.2*single_1_1 + 0.4*single_2_1

'''
加权相加 0.7345
'''
# first_half = 0.4*double_2_0 + 0.2*single_1_0 + 0.4*single_2_0
# final_half = 0.4*double_2_1 + 0.2*single_1_1 + 0.4*single_2_1

'''
加权相加 0.7345和上面分一样
'''
# first_half = 0.3*double_2_0 + 0.3*single_1_0 + 0.4*single_2_0
# final_half = 0.3*double_2_1 + 0.3*single_1_1 + 0.4*single_2_1



'''
加权相加 0.730
'''
# first_half = 0.3*double_0 + 0.4*single_1_0 + 0.4*single_2_0
# final_half = 0.3*double_1 + 0.4*single_1_1 + 0.4*single_2_1


'''
加权相加 0.7311
'''
# first_half = 0.3*double_0 + 0.3*single_1_0 + 0.4*single_2_0
# final_half = 0.3*double_1 + 0.3*single_1_1 + 0.4*single_2_1


'''
加权相加 0.73194
'''
# first_half = 0.4*double_0 + 0.1*single_1_0 + 0.5*single_2_0
# final_half = 0.4*double_1 + 0.1*single_1_1 + 0.5*single_2_1


'''
加权相加 0.7340
'''
# first_half = (0.25-0.1)*double_0 + (0.25+0.1)*double_2_0 +  (0.25-0.1)*single_1_0 + (0.25+0.1)*single_2_0
# final_half = (0.25-0.1)*double_1 + (0.25+0.1)*double_2_1 + (0.25-0.1)*single_1_1 + (0.25+0.1)*single_2_1

'''
加权相加 0.7340
'''
# first_half = (0.25-0.1)*double_0 + (0.25+0.1)*double_2_0 +  (0.25-0.1)*single_1_0 + (0.25+0.1)*single_2_0
# final_half = (0.25-0.1)*double_1 + (0.25+0.1)*double_2_1 + (0.25-0.1)*single_1_1 + (0.25+0.1)*single_2_1


'''
softmax一下再加,0.60多效果不行
'''
# first_half = double_0_softmax + single_1_0_softmax
# final_half = double_1_softmax + single_1_1_softmax

final_array = np.concatenate((first_half,final_half),axis=0)
predictions = np.argmax(final_array,axis=1)

from category_id_map import lv2id_to_category_id
import pandas as pd
# 1. load data

from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import torch
from torch.utils.data import SequentialSampler, DataLoader
from time import strftime
from config import parse_args

from category_id_map import lv2id_to_category_id,category_id_to_lv2id

import time
import json
import pandas as pd
import datetime
import time
from time import strftime
from tqdm import tqdm
import argparse
from sklearn.model_selection import StratifiedKFold



args = parse_args()


with open(args.train_annotation, 'r', encoding='utf8') as f:
    json_anns = json.load(f)
anns = pd.DataFrame(json_anns)
anns['raw_index'] = [x for x in range(len(anns))]
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=args.fold, random_state=2022, shuffle=True)
for i, (train_index, vali_index) in enumerate(kf.split(anns, anns.category_id)):
    train_data = anns.iloc[train_index, :].reset_index(drop=True)
    val_data = anns.iloc[vali_index, :].reset_index(drop=True)
    
    labels = []
    for i in val_data.category_id.to_list():
        labels.append(category_id_to_lv2id(i))
    results = evaluate(predictions, labels)        
    print(results)   
    break
