#%%writefile pretrain.py
import os, math, random, time, sys, gc,  sys, json, psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
from time import strftime

import numpy as np
import pandas as pd
from transformers import get_cosine_schedule_with_warmup

from pretrain_cfg import *

from model_pretrain import *

from util import *
from functools import partial

from config import parse_args
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from transformers import get_cosine_schedule_with_warmup
from data_helper import *
gc.enable()
    
DEVICE = 'cuda'           
def reduce_mean(tensor, nprocs):
    nprocs = 2
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) #数据规约，先对loss求和，再除GPU数
    rt /= nprocs
    return rt


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
    
    masked_lm_loss, loss_ita, loss_itm  = model(input_ids, attention_mask,video_feature, video_mask)
    return masked_lm_loss, loss_ita, loss_itm 


from tqdm import tqdm
def train(model, train_loader,train_sampler,optimizer, get_pred_and_loss,batch_size,scheduler=None, num_epochs=5,device=0):
    best_val_loss, best_epoch, step = None, 0, 13750
    start = time.time()
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    #ema = EMA(model, 0.999) #历史200个step的指数平均
    #ema.register()
    cnt = 0
    cnt2 = 0
    
    for epoch in tqdm(range(num_epochs)):
        train_loss = AverageMeter()
        mlm_losses = AverageMeter()
        itm_losses = AverageMeter()
        itc_losses = AverageMeter()
        if epoch <= 0:
            continue
        pbar = tqdm(train_loader)
        for item in pbar:
            model.train()
            optimizer.zero_grad()
            with autocast():
                masked_lm_loss, loss_ita, loss_itm  = get_pred_and_loss(model, item)
                loss = masked_lm_loss + loss_ita + loss_itm
            #同步等待另一张卡前向计算完
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)  #把loss平均一下
            scaler.scale(reduced_loss).backward()

            train_loss.update(loss.item(),batch_size)
            itm_losses.update(loss_itm.item(),batch_size)
            itc_losses.update(loss_ita.item(),batch_size)
            mlm_losses.update(masked_lm_loss.item(),batch_size)
            scaler.step(optimizer)
            scaler.update()
            #ema.update()
            if scheduler:
                scheduler.step()   
            step += 1
            cnt +=1
            cnt2 += 1
            pbar.set_postfix(epoch=epoch,total_loss = train_loss.avg ,itm_loss = itm_losses.avg ,itc_loss = itc_losses.avg,mlm_loss = mlm_losses.avg,lr=optimizer.param_groups[4]['lr'])
            if device == 0:
                if (step + 1) % 100 == 0 and step > 0:
                    logging.info(f"Epoch={epoch + 1}/{num_epochs} \t step={step:3} \t total_loss={train_loss.avg:6.4},itm_loss={itm_losses.avg:6.4},mlm_loss={mlm_losses.avg:6.4},itc_loss={itc_losses.avg:6.4}")
        
        #ema.apply_shadow()
        if device == 0:
            torch.save(model.state_dict(), f'./checkpoint/model_epoch_{epoch}_step_{step}_total_loss_{train_loss.avg:6.4}.pth')
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, f'model_optimizer_schduler_epoch_{epoch}')
        break

        
        
        
import argparse
def parse_args_for_ddp():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    return parser.parse_args()
if __name__ == "__main__":
    args_ddp = parse_args_for_ddp()
    dist.init_process_group(backend='nccl')
    # 提升速度，主要对input shape是固定时有效，如果是动态的，耗时反而慢
    cudnn.benchmark = True
    print('this is the gpu local_rank:',args_ddp.local_rank)
    args_ddp.nprocs = torch.cuda.device_count()
    device = args_ddp.local_rank    
    torch.cuda.set_device(args_ddp.local_rank)  #设置当前cuda是几号设备
    
    
    args = parse_args()
    args.local_rank = args_ddp.local_rank
    args.nprocs = args_ddp.nprocs
    #batch_size = BATCH_SIZE // args_ddp.nprocs  #为了均衡负载每个GPU的batch数
    args.seed += 1
    setup_seed(args)
    if  device == 0:
        print('分布式训练后端启动成功')
        print('nprocs:',args.nprocs)
        setup_logging_continue(args)


    #### 把100w无标注的和10w有标注的利用pandas把annation合并到一起 ####
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
        #logging.info(f'============================训练集的样本数:{len(df)}==================================')
        
    if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    #创建dataset and dataloader
    train_dataset = MultiModalDataset(args,df)
    #分布式环境的sampler and dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,seed=2022,drop_last=True)
    
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=BATCH_SIZE // args_ddp.nprocs,
                                        sampler=train_sampler,
                                        drop_last=True)
    total_steps = NUM_EPOCHS * len(train_dataloader.dataset)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    
    
    model = Clip_Vit_Cross_Model_Pretrain(args.bert_dir,args.num_top_layer).cuda()
    state = torch.load('./model_optimizer_schduler_epoch_0')
    new_state = {}
    for key in state['net'].keys():
        new_state[key.replace('module.','')] = state['net'][key]
    model.load_state_dict(new_state)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device],find_unused_parameters=True)
    
    optimizer, scheduler = build_optimizer_continue(args, model, len(train_dataloader))
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    del state,new_state
    import gc 
    gc.collect()
    train(model,train_dataloader,train_sampler,optimizer,
                             get_pred_and_loss=get_pred_and_loss,
                             scheduler=scheduler, num_epochs=NUM_EPOCHS,batch_size=BATCH_SIZE// args_ddp.nprocs,device=device)
