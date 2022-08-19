#%%writefile pretrain.py
import os, math, random, time, sys, gc,  sys, json, psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
# from imp import reload
# reload(logging)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     #filemode="w",
#     datefmt='%H:%M:%S',
#     filename = './pretrain.log',
#     filemode = 'w'
# )
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import numpy as np
import pandas as pd
from transformers import get_cosine_schedule_with_warmup
from config.data_cfg import *
from config.model_cfg import *
from config.pretrain_cfg import *
from data.record_trans import record_transform
#from data.qq_dataset import QQDataset
from qqmodel.qq_uni_model import QQUniModel
from optim.create_optimizer import create_optimizer,build_optimizer,build_optimizer_continue
from utils.eval_spearman import evaluate_emb_spearman
from utils.utils import set_random_seed,setup_logging,setup_logging_continue
from functools import partial
from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset
from config.config import parse_args
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ChainDataset
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup
from data.data_helper import *
gc.enable()
import argparse
from transformers import AdamW,get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
               

class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
def get_pred_and_loss(model, item, task=None):
    video_feature = item['frame_input'].cuda()
    input_ids = item['title_input'].cuda()
    attention_mask = item['title_mask'].cuda()
    video_mask = item['frame_mask'].cuda()
    text_tokne_type = item['text_token_type'].cuda()
    video_token_type = item['video_token_type'].cuda()
    
    target = None
    if 'target' in item:
        target = item['target'].to(DEVICE)
    
    pred, loss,itm_loss,masked_mfm_loss,masked_lm_loss,itc_loss = model(video_feature, video_mask, input_ids, attention_mask, text_tokne_type,video_token_type,target, task)
    return pred, loss,itm_loss,masked_mfm_loss,masked_lm_loss,itc_loss

def reduce_mean(tensor, nprocs):
    nprocs = 2
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) #数据规约，先对loss求和，再除GPU数
    rt /= nprocs
    return rt

from tqdm import tqdm
def train(model, train_loader,optimizer,pretrain_sampler, get_pred_and_loss,batch_size,scheduler=None, num_epochs=5,device=None):
    best_val_loss, best_epoch, step = None, 0, 137496
    start = time.time()
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    #ema = EMA(model, 0.999) #历史200个step的指数平均
    #ema.register()
    cnt = 0
    cnt2 = 0
    
    # if device==0:
    #     logging.info(f"total step={num_epochs*len(train_loader)}\t one epoch step={len(train_loader)}")
    for epoch in range(num_epochs):
        train_loss = AverageMeter()
        mlm_losses = AverageMeter()
        itm_losses = AverageMeter()
        mfm_losses = AverageMeter()
        itc_losses = AverageMeter()
        if epoch <= 5:
            continue
        pbar = tqdm(train_loader)

        for item in pbar:
            model.train()
            optimizer.zero_grad()
            with autocast():
                pred,loss,itm_loss,masked_mfm_loss,masked_lm_loss,itc_loss = get_pred_and_loss(model, item)
            
            train_loss.update(loss.item(),batch_size)
            mlm_losses.update(masked_lm_loss.item(),batch_size)
            itm_losses.update(itm_loss.item(),batch_size)
            # mfm_losses.update(masked_mfm_loss.item(),batch_size)
            itc_losses.update(itc_loss.item(),batch_size)
            #同步等待另一张卡前向计算完
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)  #把loss平均一下
            scaler.scale(reduced_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #ema.update()
            if scheduler:
                scheduler.step()   
            step += 1
            cnt +=1
            cnt2 += 1
            pretrain_sampler.set_epoch(step)
            #if device==0:
            # pbar.set_postfix(device = device,epoch=epoch,total_loss = train_loss.avg ,itm_loss = itm_losses.avg ,mfm_loss = mfm_losses.avg,mlm_loss = mlm_losses.avg ,itc_loss = itc_losses.avg,lr=optimizer.param_groups[4]['lr'])
            pbar.set_postfix(device = device,epoch=epoch,total_loss = train_loss.avg ,itm_loss = itm_losses.avg,mlm_loss = mlm_losses.avg ,itc_loss = itc_losses.avg,lr=optimizer.param_groups[4]['lr'])
            if device==0:
                if (step + 1) % 500 == 0 and step > 0:
                    # logging.info(f"Epoch={epoch + 1}/{num_epochs} \t step={step:3} \t total_loss={train_loss.avg:6.4},itm_loss={itm_losses.avg:6.4},mlm_loss={mlm_losses.avg:6.4},mfm_loss={mfm_losses.avg:6.4},itc_loss={itc_losses.avg:6.4},lr[0]={optimizer.param_groups[0]['lr']}")
                    logging.info(f"Epoch={epoch + 1}/{num_epochs} \t step={step:3} \t total_loss={train_loss.avg:6.4},itm_loss={itm_losses.avg:6.4},mlm_loss={mlm_losses.avg:6.4},itc_loss={itc_losses.avg:6.4},lr[0]={optimizer.param_groups[0]['lr']}")
        #ema.apply_shadow()
        if device==0:
            torch.save(model.state_dict(), f'./checkpoint/model_epoch_{epoch}_step_{step}_total_loss_{train_loss.avg:6.4}.pth')
            
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, f'model_optimizer_schduler_epoch_{epoch}')
        #ema.restore()
if __name__ == "__main__":
    set_random_seed(SEED+6) #是因为前面跑了6个epoch停了
    args = parse_args()
    dist.init_process_group(backend='nccl')
    # 提升速度，主要对input shape是固定时有效，如果是动态的，耗时反而慢
    cudnn.benchmark = True
    print('this is the gpu local_rank:',args.local_rank)
    args.nprocs = torch.cuda.device_count()
    device = args.local_rank    
    torch.cuda.set_device(args.local_rank)  #设置当前cuda是几号设备
    batch_size = BATCH_SIZE // args.nprocs  #为了均衡负载每个GPU的batch数
    
    
    if args.local_rank==0:
        setup_logging_continue(args)
        print('分布式训练后端启动成功')
        print('nprocs:',args.nprocs)

    #### 把100w无标注的和10w有标注的利用pandas把annation合并到一起 ####
    if args.local_rank==0:
        print("Creating dataset")
        #logging.info("Creating dataset")
    
    with open(args.train_annotation, 'r', encoding='utf8') as f:
        json_anns = json.load(f)
    train_df = pd.DataFrame(json_anns)
    train_df['raw_index'] = [x for x in range(len(train_df))]
    
    
    with open(args.unlabeld_annotation, 'r', encoding='utf8') as f:
        json_anns = json.load(f)
    unlabeld_df = pd.DataFrame(json_anns)
    unlabeld_df['raw_index'] = [x for x in range(len(unlabeld_df))]
    
    train_df.drop(['category_id'],axis=1,inplace=True)
    train_df['is_train'] = 1
    unlabeld_df['is_train'] = 0
    print(unlabeld_df.shape)
    print(train_df.shape)
    df = pd.concat((unlabeld_df,train_df)).reset_index(drop=True)
    if  device == 0:
        print(f'============================训练集的样本数:{len(df)}==================================')

    if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    #创建dataset and dataloader
    train_dataset = MultiModalDataset(args,df)
    #分布式环境的sampler and dataloader
    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,seed=2022,drop_last=True)
    
    pretrain_dataloader = dataloader_class(train_dataset,
                                        batch_size=BATCH_SIZE // args.nprocs,
                                        sampler=pretrain_sampler,
                                        drop_last=True)
    
    
    


    
    #### Model ####
    if args.local_rank==0:
        print("Creating model")
        logging.info("Creating model")
    model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK).cuda()
    state = torch.load('./model_optimizer_schduler_epoch_5')
    model.load_state_dict(state['net'])
    optimizer,scheduler = build_optimizer_continue(args, model,len(pretrain_dataloader))
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    
    train(model,pretrain_dataloader,optimizer,pretrain_sampler,
                            get_pred_and_loss=get_pred_and_loss,
                            scheduler=scheduler, num_epochs=NUM_EPOCHS,batch_size=BATCH_SIZE//2,device=device)
