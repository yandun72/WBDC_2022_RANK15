import os
import io
import json
import torch
import zipfile
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
import datetime
import time
from time import strftime
import swin
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import pandas as pd
from functools import partial
import torch.utils.data.distributed
import os
import io
import json
import torch
from clip_model_offical import build_model
import zipfile
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import pandas as pd
from functools import partial
import torch.utils.data.distributed
import random

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
def parse_args():
    parser = argparse.ArgumentParser("Visual feature extraction")
    parser.add_argument('--zip_frame_dir', type=str, default='/home/tione/notebook/data/zip_frames/labeled/')
    parser.add_argument('--ann_path', type=str, default='/home/tione/notebook/data/annotations/labeled.json')
    parser.add_argument('--swin_pretrained', type=str, default='opensource_models/swin_tiny_patch4_window7_224.pth')
    parser.add_argument('--output_path', type=str, default='/home/tione/notebook/data/zip_feats/labeled.zip')
    args = parser.parse_args()
    return args

class Config():
    def __init__(self):
        self.zip_frame_dir = '/opt/ml/input/data/zip_frames/test/'
        self.ann_path = '/opt/ml/input/data/annotations/test.json'
        
        self.output_path_0 = './labeled_0.zip'
        self.output_path_1 = './labeled_1.zip'
        self.max_video_frames = 15
        self.num_workers = 8
        self.batch_size = 100
        self.clip_vit_dir = './inference_model/ViT-B-16.pt'

    
import argparse
def parse_args_for_ddp():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    return parser.parse_args()


    
class RawFrameDataset(Dataset):

    def __init__(self,
                 ann: str,
                 zip_frame_dir: str,
                 max_video_frames: int = 15):
        """ This class is used to load raw video frames.
        Args:
            ann_paths (str): the annotation file path.
            zip_frame_dir (str): the directory that saves zip frames.
            max_video_frames (str): the maximum number of video frames.
        """
        # # load annotations
        # with open(ann_path, 'r', encoding='utf8') as f:
        #     self.anns = json.load(f)
        self.anns = ann
        self.zip_frame_dir = zip_frame_dir
        self.max_video_frames = max_video_frames
        
        # we follow the common practice as in the ImageNet's preprocessing.
        self.transform = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> dict:
        return len(self.anns)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Extract the frame tensor from zipped file.
        The output tensor is in shape of [MAX_FRAMES, 3, 224, 224]
        """
        feedid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, feedid[-3:], f'{feedid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        img_name_list = handler.namelist()
        img_name_list = sorted(img_name_list)
        
        select_inds = list(range(len(img_name_list)))
        random.shuffle(select_inds)
        select_inds = sorted(select_inds[:self.max_video_frames])
        
        img_name_list_2 = []
        for i in select_inds:
            img_name_list_2.append(img_name_list[i])
        img_name_list_2 = img_name_list_2
        img_tensor = torch.zeros(self.max_video_frames, 3, 224, 224)
        for i, img_name in enumerate(img_name_list_2):
            i_img_content = handler.read(img_name)
            i_img = Image.open(io.BytesIO(i_img_content))
            i_img_tensor = self.transform(i_img)
            img_tensor[i, ...] = i_img_tensor
        handler.close()
        num_frames = torch.LongTensor([len(img_name_list_2)])
        return dict(img=img_tensor, num_frames=num_frames)




def main():
    now=datetime.datetime.now()
    seed_everything(1998)
    args_ddp = parse_args_for_ddp()
    dist.init_process_group(backend='nccl')
    # 提升速度，主要对input shape是固定时有效，如果是动态的，耗时反而慢
    cudnn.benchmark = True
    print('this is the gpu local_rank:',args_ddp.local_rank)
    args_ddp.nprocs = torch.cuda.device_count()
    device = args_ddp.local_rank    
    torch.cuda.set_device(args_ddp.local_rank)  #设置当前cuda是几号设备
    
    args = Config()
    
    print('加载clip的vit权重')
    checkpoint_dict = torch.load(args.clip_vit_dir).state_dict()
    clip_model = build_model(checkpoint_dict)
    model = clip_model.visual
    
    
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(),device_ids=[device],find_unused_parameters=True)
    model.eval()
    
    with open(args.ann_path, 'r', encoding='utf8') as f:
        json_anns = json.load(f)
    
    
    if device == 0:
        anns = json_anns[0:len(json_anns)//2]
        
    elif device == 1:
        anns = json_anns[len(json_anns)//2:]

    dataset = RawFrameDataset(anns, args.zip_frame_dir)
    
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    dataloader = dataloader_class(dataset,batch_size=args.batch_size,drop_last=False,shuffle=False)        

    if device == 0:
        output_path = args.output_path_0
    elif device == 1:
        output_path = args.output_path_1
    
    assert not os.path.isfile(output_path), f"{output_path} already exists. " \
                                                  "If you want to override it, please manually delete this file."
    output_handler = zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    with torch.no_grad():
        cur = 0
        #pbar = tqdm(dataloader)
        for dataitem in dataloader:
            img, num_frames = dataitem['img'], dataitem['num_frames']
            B, L = img.shape[0:2]
            img = img.view((B * L, ) + img.shape[2:])
            with autocast():
                feature = model(img)
            feature = feature.view(B, L, -1)
            feature = feature.cpu().numpy().astype(np.float16)
            # print(num_frames)
            # print(feature.shape)
            #pbar.set_postfix(device=device)
            #c = d
            for i in range(B):
                feedid = dataset.anns[cur]['id']
                ioproxy = io.BytesIO()
                np.save(ioproxy, feature[i, :int(num_frames[i])])
                npy_str = ioproxy.getvalue()
                output_handler.writestr(f'{feedid}.npy', npy_str)
                cur += 1
                
                if cur % 1000 == 0:
                    if device == 0:
                        print(f"Extract feature {cur}/{len(dataset)}")
    output_handler.close()
    end=datetime.datetime.now()
    if device == 0:
        print(f'抽帧花费的时间:{(end-now).seconds}秒')

if __name__ == '__main__':
    main()
