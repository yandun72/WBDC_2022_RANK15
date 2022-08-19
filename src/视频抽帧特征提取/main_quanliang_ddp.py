import pandas as pd
import json
import random
from collections import OrderedDict
import zipfile
from io import BytesIO
from functools import partial
from sklearn.model_selection import StratifiedKFold

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
import random
from config import parse_args
from data_helper import *

from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import warnings

warnings.filterwarnings("ignore")
import logging
import os
import time
from sklearn.utils import shuffle
import torch
import random
from config import parse_args
from data_helper import  MultiModalDataset
from model_albef import *
import json
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import numpy as np
from tqdm import tqdm

print('*********************************no fgm*******************************************************')


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


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


from tqdm import tqdm
from model_albef import ALBEF

def freeze_or_not_swin(model,requires_grad=False):
    for name ,param in model.named_parameters():
        # if 'visual_backbone.layers.2' in name:  #只冻结视频
        if 'visual_backbone' in name:  #只冻结视频
            param.requires_grad = requires_grad

def train_and_validate(args, train_dataloader):
    # 1. load data
    # 2. build model and optimizers
    # model = VLModel(args)
    model = ALBEF(text_encoder=args.bert_dir)
    
    
    optimizer, scheduler = build_optimizer(args, model, len(train_dataloader))
    freeze_or_not_swin(model,requires_grad=False)
    model = torch.nn.parallel.DataParallel(model.to(args.device))
    ema3 = EMA(model, 0.999)  # 历史200个step的指数平均   
    ema3.register()
    
    fgm = FGM(model, 'word_embeddings.')
    
    # 3. training
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs

    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    #swa_model = torch.optim.swa_utils.AveragedModel(model)
    
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
            with autocast():
                loss_first, accuracy_first, _, _ = model(batch)
                loss_first = loss_first.mean()
                accuracy_first = accuracy_first.mean()
                loss = loss_first
            scaler.scale(loss).backward()

            with autocast():
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                loss_adv, accuracy2, _, _ = model(batch)
                loss_adv = loss_adv.mean()/2.0
            scaler.scale(loss_adv).backward()
            fgm.restore()

            losses.update(loss.item(), args.batch_size)
            acc1.update(accuracy_first.item(), args.batch_size)
            
            
            scaler.step(optimizer)
            # optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            scaler.update()

            ema3.update()
            
            step += 1
            torch.cuda.empty_cache()
            pbar.set_postfix(epoch=epoch)
            
            if step % args.print_steps == 0:
                print(
                    f"Epoch {epoch} step {step} lr {optimizer.param_groups[0]['lr']}: total loss {losses.avg:.3f},accuracy1 {acc1.avg:.3f} ")
                logging.info(
                    f"Epoch {epoch} step {step} lr {optimizer.param_groups[0]['lr']}: total loss {losses.avg:.3f},accuracy1 {acc1.avg:.3f}")
        ema3.apply_shadow()
        state_dict = {k: v for k, v in model.module.state_dict().items() if 'relative_positions' not in k}
        torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                    f'{args.savedmodel_path}/model_macbert_quanliang.bin')
        ema3.restore()

import argparse
def parse_args_for_ddp():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    return parser.parse_args()

args = parse_args()
setup_logging(args)
setup_device(args)
seed_everything(42)
logging.info("Training/evaluation parameters: %s", args)
with open(args.train_annotation, 'r', encoding='utf8') as f:
    json_anns = json.load(f)
anns = pd.DataFrame(json_anns)
train_dataset = MultiModalDataset(args, args.train_zip_feats, anns)

if args.num_workers > 0:
    dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers)
else:
    dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
        
train_dataloader = dataloader_class(train_dataset,
                                        shuffle=True,
                                        batch_size=args.batch_size,
                                        drop_last=True)

train_and_validate(args, train_dataloader)


