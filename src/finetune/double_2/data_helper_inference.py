import json
import random
import zipfile
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

from io import BytesIO
from functools import partial
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import os
from category_id_map import category_id_to_lv2id, CATEGORY_ID_LIST


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 zip_feat_path,
                 data,
                 test_mode: bool = False):
        self.args = args
        
        
            
        self.zip_feat_path = zip_feat_path
        if test_mode == False:
            self.max_frame = args.max_frames
            self.bert_seq_length = args.bert_seq_length
        else:
            self.max_frame = args.max_frames_infer
            self.bert_seq_length = args.bert_seq_length_infer
        self.test_mode = test_mode
        self.num_workers = args.num_workers
             
        self.data = data
        self.raw_data_len = 10000
        self.id = data.id.to_list()
        self.title = data.title.to_list()
        self.asr = data.asr.to_list()
        self.raw_index = data.raw_index.to_list()
        self.ocr = data.ocr.to_list()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True,never_split = ['[unused1]','[unused2]','[unused3]','[unused4]'])
        if args.vision_model != 'swin_v2':
            self.transform = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = Compose([
                Resize(256),
                CenterCrop(192),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])  
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handle_1 = [None for _ in range(args.num_workers)]
            self.handle_0 = [None for _ in range(args.num_workers)]
        else:
            self.handle_1 = zipfile.ZipFile('./labeled_1.zip', 'r')
            self.handle_0 = zipfile.ZipFile('./labeled_0.zip', 'r')
    def __len__(self) -> int:
        return len(self.data)

    def get_visual_frames(self, idx: int) -> tuple:
        vid = self.id[idx]
        raw_index = self.raw_index[idx]


        #print(vid,idx,raw_index,self.data_len,zip_path)
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handle_0[worker_id] is None:
                self.handle_0[worker_id] = zipfile.ZipFile('./labeled_0.zip', 'r')
                #self.handle_0[worker_id] = zipfile.ZipFile('/home/tione/notebook/data/Vit_B_16_unlabeled_zip_feats/labeled_0.zip', 'r')
            if self.handle_1[worker_id] is None:
                self.handle_1[worker_id] = zipfile.ZipFile('./labeled_1.zip', 'r')
                #self.handle_1[worker_id] = zipfile.ZipFile('/home/tione/notebook/data/Vit_B_16_unlabeled_zip_feats/labeled_1.zip', 'r')
            handle_0 = self.handle_0[worker_id]
            handle_1 = self.handle_1[worker_id]
        else:
            handle_1 = self.handle_1
            handle_0 = self.handle_0
        
        if raw_index < 25000//2:
            try:
                raw_feats = np.load(BytesIO(handle_0.read(name=f'{vid}.npy')), allow_pickle=True)
            except Exception as e:
                raw_feats = np.load(BytesIO(handle_1.read(name=f'{vid}.npy')), allow_pickle=True)
        else:
            try:
                raw_feats = np.load(BytesIO(handle_1.read(name=f'{vid}.npy')), allow_pickle=True)
            except Exception as e:
                raw_feats = np.load(BytesIO(handle_0.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape
        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)

        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
            
        else:
            # randomly sample when test mode is False
            select_inds = list(range(num_frames))
            random.shuffle(select_inds)
            select_inds = select_inds[:self.max_frame]
            select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.from_numpy(feat)
        mask = torch.LongTensor(mask)
        return feat, mask
    

    
    def tokenize_text(self, text: str) -> tuple:#文本
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_frames(idx)

        # Step 2, load title tokens
        ocr_text = []
        for o in self.ocr[idx]:
            ocr_text.append(o['text'])
        ocr_text=''.join(ocr_text)

        #todo
        all_text = '[unused4]'+self.title[idx] +'[unused1]'+ self.asr[idx] + '[unused3]' + ocr_text
        title_input, title_mask = self.tokenize_text(all_text)
        text_token_type = torch.LongTensor([0]*len(title_input))
        
        video_token_type = torch.LongTensor([1]*len(frame_input))
        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask,
            text_token_type=text_token_type,
            video_token_type=video_token_type
        )
        if not self.test_mode:
            label = category_id_to_lv2id(self.data.category_id.to_list()[idx])
            data['label'] = torch.LongTensor([label])
        return data
    
