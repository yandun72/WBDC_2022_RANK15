import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
#from  bert_model import BertCrossLayer, BertAttention
#from  swin_transformer as swin
# from  heads import Pooler
# from  objectives import init_weights

#from clip_model_offical import build_model

from qq_uni_model import *
import datetime

class Position_Embedding_For_Video(nn.Module):
    def __init__(self,maxframe,model):
        super().__init__()
        self.position_embeddings = model.video_position_embeddings.position_embeddings
        self.LayerNorm = model.video_position_embeddings.LayerNorm
        self.maxframe = maxframe
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("position_ids", torch.arange(16).expand((1, -1)))
    def forward(self, embeddings):
        position_ids = self.position_ids[:, 0 : self.maxframe]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class Single_Model_Finetune(nn.Module):
    def __init__(self,args,is_inference=False):
        super().__init__()
        now=datetime.datetime.now()
        if is_inference == False:
            pretrain_state = torch.load('../../pretrain/single_2/checkpoint/model_epoch_2.pth')
            new_state = {}
            for key in pretrain_state.keys():
                if 'module.' not in key:
                    new_state[key] = pretrain_state[key]
                else:
                    new_state[key.replace('module.','')] = pretrain_state[key]
            model = Single_Stream_Model_Pretrain(args.bert_dir)
            model.load_state_dict(new_state,strict=False)
        else:
            model = Single_Stream_Model_Pretrain(args.bert_dir)
            
        self.text_encoder = model.text_encoder
        self.video_position_embeddings = Position_Embedding_For_Video(args.max_frames,model)
        self.video_transform_2 = model.video_transform_2
        #================下游分类 and 辅助参数======================
        self.classifier = nn.Linear(768*3, 200)
        self.gelu = nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self,item,inference=False):
        frame_input = item['frame_input'].cuda()
        title_input = item['title_input'].cuda()
        title_mask = item['title_mask'].cuda()
        frame_mask = item['frame_mask'].cuda()
        if inference == False:
            label = item['label'].cuda()
        frame_input = self.video_position_embeddings(frame_input) #为视频的帧添加位置信息
        frame_input_normal = self.video_transform_2(frame_input)
        text_output = self.text_encoder.bert.embeddings(input_ids=title_input) #只用embedding信息做itc的loss计算
        embedding_input = torch.cat([frame_input_normal, text_output], 1)
        
        mask = torch.cat([frame_mask, title_mask], 1)
        mask2 = mask.clone()
        video_mask = frame_mask.clone()
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        encoder_outputs = self.text_encoder.bert.encoder(embedding_input, attention_mask=mask)[0]
        B,M,D = frame_input.shape
        
        video_embedding = (encoder_outputs[:,0:M,:] * video_mask.unsqueeze(-1)).sum(1) / video_mask.sum(1).unsqueeze(-1)
        text_embedding = (encoder_outputs[:,M:,:] * title_mask.unsqueeze(-1)).sum(1) / title_mask.sum(1).unsqueeze(-1)
        text_video_outputs = (encoder_outputs * mask2.unsqueeze(-1)).sum(1) / mask2.sum(1).unsqueeze(-1)
        output = torch.cat((text_embedding,video_embedding,text_video_outputs),dim=-1)
        prediction = self.classifier(self.dropout(output))
        if inference:  # 预测
            return torch.argmax(prediction, dim=1),prediction
        else:
            loss, accuracy, pred_label_id, label = self.cal_loss(prediction, label)
            return loss, accuracy, pred_label_id, label

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label