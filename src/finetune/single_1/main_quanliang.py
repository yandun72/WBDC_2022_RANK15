import pandas as pd
import json
import random
from collections import OrderedDict
import zipfile
from io import BytesIO
from functools import partial
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from tricks import *
##
import os
import random
import logging
import os
import time
import torch
import numpy as np
import random
from config import parse_args
from data_helper import create_dataloaders
from model import VLModel
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from model_cfg import *
from pretrain_cfg import *
import warnings
warnings.filterwarnings("ignore")
import logging
import os
import time
from sklearn.utils import shuffle
import torch
import random
from config import parse_args
from data_helper import create_dataloaders,MultiModalDataset
from model import VLModel
import json
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import numpy as np
from tqdm import tqdm
import datetime
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
from time import strftime

def reduce_mean(tensor, nprocs):
    nprocs = 2
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) #数据规约，先对loss求和，再除GPU数
    rt /= nprocs
    return rt


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


def validate(model, val_dataloader,validation_sampler,device):
    model.eval()
    predictions = []
    labels = []
    losses = []
    step = 0
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    if device == 0:
        print('======================开始验证的时刻====================================')
        now=datetime.datetime.now()
        print(f'=========================={now.strftime("%Y-%m-%d %H:%M:%S")}==========================================')

    with torch.no_grad():
        for batch in val_dataloader:
            with autocast():
                loss, _, pred_label_id, label = model(batch)
            #====================同步等待另一张卡前向计算完==========================
            torch.distributed.barrier()
            if device == 1: #=================向主进程发送预测和label=============
                # print('this is gpu:',device)
                # print('I am sending:')
                # print('pred:',pred_label_id)
                # print('label:',label)
                lens_send = len(pred_label_id)
                torch.distributed.send(torch.LongTensor([lens_send]).cuda(),dst=0)
                torch.distributed.send(pred_label_id,dst=0)
                torch.distributed.send(label,dst=0)
            else: #=====================主进程负责接收其他进程发来的数据===================
                # print('this is gpu:',device)
                # print('I am waiting recving:')
                
                #===========================================================================
                #==================先用一个元素接收发送方本次send的长度========================
                lens_send = torch.LongTensor([-100]).cuda()
                torch.distributed.recv(lens_send,src=1)
                
                #===========================================================================
                #==================然后就可以创建一个定长的缓冲区用来接收发送的数据了============
                pred_label_id_from_device_1 = torch.LongTensor([-100]*lens_send.cpu().numpy()[0]).cuda()
                torch.distributed.recv(pred_label_id_from_device_1,src=1)
                
                label_from_device_1 = torch.LongTensor([-100]*lens_send.cpu().numpy()[0]).cuda()
                torch.distributed.recv(label_from_device_1,src=1)
                
                #===============================================================================
                # print('recved!!!')
                # print('pred:',pred_label_id)
                # print('label:',label)
            #loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            if device == 0:
                #===========================================================================
                #=======主进程负责将其他进程预测的结果和label全部弄到一个变量去进行计算分数=======
                predictions.extend(pred_label_id_from_device_1.cpu().numpy())
                labels.extend(label_from_device_1.cpu().numpy())
            #===============================================================================
            #========================下面这步很重要，保证次序不会乱============================
            validation_sampler.set_epoch(step)
            step += 1
    #loss = sum(losses) / len(losses)
    if device == 0:
        end=datetime.datetime.now()
        print('======================结束验证的时刻====================================')
        print(f'=========================={end.strftime("%Y-%m-%d %H:%M:%S")}==========================================')
        print(f'花费的时间:{(end-now).seconds/60.0:.2f}分钟')
    if device == 0:
        # print('this is gpu:',device)
        # print('the final pred and label is:')
        # print('pred:',predictions)
        # print('label:',labels)
        results = evaluate(predictions, labels)
        model.train()
        return results
    else:
        return None


def train_and_validate(args, train_dataloader, train_sampler,fold,device):
    # 1. load data
    # 2. build model and optimizers
    # model = VLModel(args)
    model = VLModel(args).cuda()
   
    print('Freeze')
    for name ,param in model.named_parameters():
        # if 'visual_backbone' in name or 'cross_modal' in name:
        #     print(name)
        if 'visual_backbone' in name and args.freeze == True:  #只冻结视频 
            param.requires_grad = False
        if f'encoder.layer.layer.{args.freeze_bert_layer}' in name:
            param.requires_grad = False
      
    optimizer, scheduler = build_optimizer(args, model, len(train_dataloader))
    
    
    
    if args.fgm == True:
        fgm = FGM(model, 'word_embeddings.')
        
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device],find_unused_parameters=True)
    ema3 = EMA(model, 0.999)  # 历史200个step的指数平均   
    ema3.register()
    # 3. training
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = (len(train_dataloader)//args.nprocs) * args.max_epochs

    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    
    for epoch in range(args.max_epochs):
        step = 0
        losses = AverageMeter()
        acc1 = AverageMeter()
        acc2 = AverageMeter()
        kl_losss = AverageMeter()
        loss_1 = AverageMeter()
        loss_2 = AverageMeter()
        total_loss = AverageMeter()
        
        pbar = tqdm(train_dataloader)
        for batch in tqdm(pbar):
            model.train()
            torch.cuda.empty_cache()
            
            with autocast():
                loss_first, accuracy_first, _, _ = model(batch)
                loss_first = loss_first.mean()
                accuracy_first = accuracy_first.mean()
                loss = loss_first / args.accumlate_step
            #scaler.scale(loss).backward()
            #同步等待另一张卡前向计算完
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)  #把loss平均一下
            scaler.scale(reduced_loss).backward()
            
            
            if args.fgm == True:
                with autocast():
                    # 对抗训练
                    fgm.attack()  # 在embedding上添加对抗扰动
                    loss_adv, accuracy2, _, _ = model(batch)
                    loss_adv = loss_adv.mean()
                torch.distributed.barrier()
                reduced_loss_adv = reduce_mean(loss_adv, args.nprocs)  #把loss平均一下
                scaler.scale(reduced_loss_adv).backward()
                #scaler.scale(loss_adv).backward()
                fgm.restore()

            losses.update(loss.item(), args.batch_size//args.nprocs)
            acc1.update(accuracy_first.item(), args.batch_size)
            if (step + 1) % args.accumlate_step ==0:
                scaler.step(optimizer)
                # optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                scaler.update()
                ema3.update()
            
            step += 1
            pbar.set_postfix(fold=fold, epoch=epoch)
            train_sampler.set_epoch(step)
            
            
            if device==0:
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    print(
                        f"Epoch {epoch} step {step} lr {optimizer.param_groups[0]['lr']}: total loss {losses.avg:.3f},accuracy1 {acc1.avg:.3f} ")
                    logging.info(
                        f"Epoch {epoch} step {step} lr {optimizer.param_groups[0]['lr']}: total loss {losses.avg:.3f},accuracy1 {acc1.avg:.3f}")
        if device==0:
            if epoch >= 2:
                ema3.apply_shadow()
                state_dict = {k: v for k, v in model.module.state_dict().items() if 'relative_positions' not in k}
                torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                           f'{args.savedmodel_path}/model_macbert_seed_2012_ema0.999_epoch_{epoch}_fold_{fold}.bin')
                ema3.restore()
        if  epoch == 3: #第四个epoch结束
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
    
    
    
    # for fold in range(5):
    #     os.makedirs(f'{args.savedmodel_path}/fold_{fold}', exist_ok=True)
    # os.makedirs(args.savedmodel_path, exist_ok=True)
    args = parse_args()
    args.local_rank = args_ddp.local_rank
    args.nprocs = args_ddp.nprocs
    batch_size = args.batch_size // args_ddp.nprocs  #为了均衡负载每个GPU的batch数
    if args.fgm == False and device == 0:
        print('*********************************no fgm*******************************************************')
    elif args.fgm == True and device == 0:
        print('*********************************has fgm*******************************************************')
    if args.local_rank==0:
        print('分布式训练后端启动成功')
        print('nprocs:',args.nprocs)
        setup_logging(args)
        logging.info("Training/evaluation parameters: %s", args)
        
    seed_everything(42)
    
    with open(args.train_annotation, 'r', encoding='utf8') as f:
        json_anns = json.load(f)
    anns = pd.DataFrame(json_anns)
    anns['raw_index'] = [x for x in range(len(anns))]
    train_dataset = MultiModalDataset(args, args.train_zip_feats, anns)
    #分布式环境的sampler and dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,seed=2022,drop_last=True)
    if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_dataloader = dataloader_class(train_dataset,
                                        sampler = train_sampler,
                                        #shuffle=True,
                                        batch_size=batch_size,
                                        drop_last=True)
    
    train_and_validate(args, train_dataloader,train_sampler,0,device)