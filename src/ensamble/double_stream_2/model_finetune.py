import torch
import torch.nn as nn
#import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from  bert_model import BertCrossLayer, BertAttention
#from  swin_transformer as swin
from  heads import Pooler
from  objectives import init_weights
#from  meter_utils import *
#from clip_model import adapt_position_encoding
from clip_model_offical import build_model
#from swin_helpers import swin_adapt_position_encoding

from model_pretrain import Clip_Vit_Cross_Model_Pretrain
from qq_uni_model import *
import datetime
class Clip_Vit_Cross_Model_finetune(nn.Module):
    def __init__(self,args,is_inference=False):
        super().__init__()
        #====================text encoder=====================
        now=datetime.datetime.now()
        if is_inference == False:
            pretrain_state = torch.load(args.pretrained_model_path)
            new_state = {}
            for key in pretrain_state.keys():
                if 'module.' not in key:
                    new_state[key] = pretrain_state[key]
                else:
                    new_state[key.replace('module.','')] = pretrain_state[key]
            model = Clip_Vit_Cross_Model_Pretrain(args.bert_dir,args.num_top_layer)
            model.load_state_dict(new_state)
        else:
            model = Clip_Vit_Cross_Model_Pretrain(args.bert_dir,args.num_top_layer)
        self.text_encoder = model.text_encoder
        self.text_transform = model.text_transform_2
        self.image_transform = model.video_transform_2
        self.token_type_embeddings = model.token_type_embeddings
        self.cross_modal_image_layers = model.cross_modal_image_layers
        
        #================下游分类 and 辅助参数======================
        self.classifier = nn.Linear(768, 200)
        #self.vision_cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.gelu = nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        
        
    def forward(self, inputs, inference=False):
        bs = inputs['frame_input'].shape[0]
        frames = inputs['frame_input'].shape[1]
        
        device = 'cuda'
        title_input = inputs['title_input'].to(device)
        title_mask = inputs['title_mask'].to(device)
        frame_input = inputs['frame_input'].to(device)
        frame_mask = inputs['frame_mask'].to(device)
        if inference == False:
            label = inputs['label'].to(device)
        
        
        #======================get text emb===============================
        text_embeds = self.text_encoder(title_input,title_mask,output_hidden_states=True).hidden_states[-1]
        torch.cuda.empty_cache()

        text_mask_shape = title_mask.shape
        extend_text_masks = self.text_encoder.get_extended_attention_mask(title_mask, text_mask_shape, device)
        #===================================将其变换到另一个空间======================================
        text_embeds = self.text_transform(text_embeds)
        

        extend_image_masks = self.text_encoder.get_extended_attention_mask(frame_mask, frame_mask.size(), device)

        #===================================将其变换到另一个空间======================================
        image_embeds = self.image_transform(frame_input)

        
        #===================在进入cross attention前加入tokentype embedding=========
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(title_mask)),
            image_embeds+ self.token_type_embeddings(torch.full_like(frame_mask, 1)),
        )
        
        
            
        for image_layer in  self.cross_modal_image_layers:
            image_embeds = image_layer(image_embeds, text_embeds, extend_image_masks, extend_text_masks)[0]

        image_feats = (image_embeds * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1).unsqueeze(-1)


        prediction = self.classifier(self.dropout(image_feats))

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