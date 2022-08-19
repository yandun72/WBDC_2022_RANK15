import torch
from torch.utils.data import SequentialSampler, DataLoader
from time import strftime
from config_inference import parse_args
from data_helper_inference import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model_inference import *
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import time
import json
import pandas as pd
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

def get_model_path_list(base_dir):
    model_lists = []
    for root, dirs, files in os.walk(base_dir):
        for _file in files:
              if 'epoch' in _file:
                    model_lists.append(os.path.join(root, _file))
    return model_lists
def inference():
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
        anns['raw_index'] = [x+12500 for x in range(len(anns))]
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
    #model_path = get_model_path_list('./inference_model')
    five_pred = []
    for i in range(5):
        now_fold_pred = []
        
        state_path = f'./inference_model/model_macbert_seed_2012_ema0.999_epoch_2_fold_{i}.bin'
        if device == 0:
            print(f'=======================fold:{i}=======================================')
            print(state_path)
        model = Clip_Vit_Cross_Model(args,is_inference=True)
        checkpoint = torch.load(state_path, map_location='cpu')
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

        with torch.no_grad():
            if device == 0:
                pbar = tqdm(dataloader)
            else:
                pbar = dataloader
            for batch in pbar:
                with autocast():
                    pred_label_id,pred_prob = model(batch, inference=True)    
                predictions.extend(pred_label_id.cpu().numpy())
                now_fold_pred.extend(pred_prob.cpu().numpy())
        now_fold_pred = np.array(now_fold_pred)
        five_pred.append(now_fold_pred)
    #print(np.array(five_pred).shape)    
    five_pred = np.argmax(np.array(five_pred).mean(axis=0),axis=1)   
    #print(five_pred.shape)
    #c = d
    #五折完了，开始求mean，生成label    
    # 4. dump results
    now=datetime.datetime.now()
    if device == 1:
        category_id_ = []
        for pred_label_id, ids in zip(five_pred, anns.id.to_list()):
            video_id = ids
            category_id = lv2id_to_category_id(pred_label_id)
            category_id_.append(category_id)
            #print(pred_label_id,category_id,'device:1')
            #f.write(f'{video_id},{category_id}\n')
        df_1 = pd.DataFrame(data={'id':anns.id.to_list(),'cat':category_id_})
        df_1.to_pickle('./result_1.pkl')
        #df_1.to_csv('./result_1.csv')
    else:
        category_id_ = []
        for pred_label_id, ids in zip(five_pred, anns.id.to_list()):
                video_id = ids
                category_id = lv2id_to_category_id(pred_label_id)
                category_id_.append(category_id)
        df_0 = pd.DataFrame(data={'id':anns.id.to_list(),'cat':category_id_})
        #df_0.to_pkl('./result_0.pkl')
        #df_0.to_csv('./result_0.csv')
    torch.distributed.barrier()
    if device == 0:
        df_1 = pd.read_pickle('./result_1.pkl')
        df = pd.concat((df_0,df_1),axis=0).reset_index(drop=True)
        df.to_csv('/opt/ml/output/result.csv',sep=',',header=0,index=False)
        #df.to_csv('./result.csv',sep=',',header=0,index=False)
        end=datetime.datetime.now()
        print(f'写入答案花费的时间:{(end-now).seconds}秒')
if __name__ == '__main__':
    inference()
