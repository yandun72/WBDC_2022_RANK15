import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from datetime import timedelta
import time
import logging
from model_finetune import Single_Model_Finetune
from config import parse_args

'''
copy from deepner
https://github.com/z814081807/DeepNER/blob/master/src/utils/functions_utils.py
'''
def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []
    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            model_lists.append(os.path.join(root, _file))
    return model_lists


def swa(model, model_dir, swa_model_dir):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)

    swa_start=0
    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            print(_ckpt)
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu'))['model_state_dict'])
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1
    torch.save(swa_model.state_dict(), swa_model_dir)


    
'''
针对使用ema了的后三个epoch模型的swa平均
'''
dir = "./save/" 
args = parse_args()
model = Single_Model_Finetune(args)
print('针对使用ema了的后三个epoch模型的swa平均')
swa(model, dir,swa_model_dir='./save/has_ema_swa_last_3epoch.bin')