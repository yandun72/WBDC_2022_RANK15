import torch
from torch.utils.data import SequentialSampler, DataLoader
from time import strftime
from config_inference import parse_args
from data_helper_inference import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model_finetune import *
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import time
import json
import pandas as pd
import os
import random
import numpy as np
import datetime
import time
from time import strftime
from tqdm import tqdm
import argparse
def parse_args_for_ddp():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    return parser.parse_args()
def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
def inference():
    seed_everything(78962)
    args = parse_args()
    args_ddp = parse_args_for_ddp()
    now=datetime.datetime.now()
    dist.init_process_group(backend='nccl')
    end=datetime.datetime.now()
    
    # 提升速度，主要对input shape是固定时有效，如果是动态的，耗时反而慢
    cudnn.benchmark = True
    print('this is the gpu local_rank:',args_ddp.local_rank)
    args_ddp.nprocs = torch.cuda.device_count()
    device = args_ddp.local_rank    
    torch.cuda.set_device(args_ddp.local_rank)  #设置当前cuda是几号设备
    
    args = parse_args()
    args.local_rank = args_ddp.local_rank
    args.nprocs = args_ddp.nprocs
    batch_size = args.batch_size   #为了均衡负载每个GPU的batch数
    
    # 1. load data
    with open(args.test_annotation, 'r', encoding='utf8') as f:
         json_anns = json.load(f)
            
    #pd.DataFrame(json_anns).to_csv('./data.csv',index=False)
    if device == 0:
        print(f'ddp启动花费的时间:{(end-now).seconds}秒')
        anns = pd.DataFrame(json_anns[0:len(json_anns)//2])
        anns['raw_index'] = [x for x in range(len(anns))]
    elif device == 1:
        anns = pd.DataFrame(json_anns[len(json_anns)//2:])
        anns['raw_index'] = [x+len(anns) for x in range(len(anns))]
    dataset = MultiModalDataset(args, args.test_zip_feats,anns, test_mode=True)
    sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers)

    # 2. load model
    now=datetime.datetime.now()
    model = Single_Model_Finetune(args,is_inference=True)
    checkpoint = torch.load('../../finetune/single_2/save/model_macbert_seed_2012_ema0.999_epoch_3_fold_0.bin', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    model = torch.nn.parallel.DistributedDataParallel(model.cuda(),device_ids=[device],find_unused_parameters=True)
    model.eval()
    end=datetime.datetime.now()
    
    if device == 0:
        print(f'导入训练的权重花费的时间:{(end-now).seconds}秒')


    # 3. inference
    predictions = []
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    len_dataloader = len(dataloader)
    now_device_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            with autocast():
                pred_label_id,pred_prob = model(batch, inference=True)    
            predictions.extend(pred_label_id.cpu().numpy())
            now_device_pred.extend(pred_prob.cpu().numpy())
    now_device_pred = np.array(now_device_pred)
    np.save(f'../single_2_device_{device}.npy',now_device_pred)
if __name__ == '__main__':
    inference()

    