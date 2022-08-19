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
        if args.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(zip_feat_path, 'r')
        self.data = data
        self.id = data.id.to_list()
        self.title = data.title.to_list()
        self.asr = data.asr.to_list()
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
        
    def __len__(self) -> int:
        return len(self.data)

    def get_visual_frames(self, idx: int) -> tuple:
        vid = self.id[idx]
        zip_path = os.path.join(self.zip_feat_path, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        if self.args.vision_model != 'swin_v2':
            frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        else:
            frame = torch.zeros((self.max_frame, 3, 192, 192), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            # randomly sample when test mode is False
            select_inds = list(range(num_frames))
            random.shuffle(select_inds)
            select_inds = select_inds[:self.max_frame]
            select_inds = sorted(select_inds)
        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask
    

    
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
    
