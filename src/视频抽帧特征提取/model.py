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
from swin import *
import datetime
class Clip_Vit_Cross_Model(nn.Module):
    def __init__(self,args,is_inference=False):
        super().__init__()
        #====================text encoder=====================
        
        #bert_config.num_hidden_layers = args.text_transforemr_layers
        #self.text_encoder = BertModel(bert_config)
        now=datetime.datetime.now()
        bert_config = BertConfig.from_pretrained(args.bert_dir)
        self.text_encoder = BertModel.from_pretrained(args.bert_dir)
#         if is_inference == False:
#             bert_config = BertConfig.from_pretrained(args.bert_dir)
#             self.text_encoder = BertModel.from_pretrained(args.bert_dir)
#         else:
#             bert_config = BertConfig()
#             bert_config.vocab_size = 21128
#             bert_config.model_type = 'bert'
            
#             self.text_encoder = BertModel(bert_config)
        
        #===================visual encoder======================
        ####################we use clip vit/b16 as our backbon###
        # self.visual_backbone = swin_base(args.swin_pretrained_path)
        # config = BertConfig()
        # config.num_hidden_layers  = 3
        # self.visual_backbone_temporal_encoder = BertModel(config).encoder
        
        
        #print('加载clip的vit权重')
        checkpoint_dict = torch.load(args.clip_vit_dir).state_dict()
        clip_model = build_model(checkpoint_dict)
        self.visual_backbone = clip_model.visual
        end=datetime.datetime.now()
        print(f'加载权重花费的时间:{(end-now).seconds}秒')
        #print(clip_model)
        #==================维度转换,先进行线性变换再Cross Attention========================
        self.text_transform = nn.Linear(768, 768)
        self.text_transform.apply(init_weights)
        self.image_transform = nn.Linear(768,768)
        self.image_transform.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, 768)
        self.token_type_embeddings.apply(init_weights)
        
        #==================cross encoder========================
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(args.num_top_layer)])
        self.cross_modal_image_layers.apply(init_weights)
        #self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(args.num_top_layer)])
        #self.cross_modal_text_layers.apply(init_weights)

        # self.cross_modal_image_pooler = Pooler(768)
        # self.cross_modal_image_pooler.apply(init_weights)
        # self.cross_modal_text_pooler = Pooler(768)
        # self.cross_modal_text_pooler.apply(init_weights)
        
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
        text_embeds = self.text_encoder(title_input,title_mask)[0]
        torch.cuda.empty_cache()
        #print(text_embeds.shape)
        text_mask_shape = title_mask.shape
        extend_text_masks = self.text_encoder.get_extended_attention_mask(title_mask, text_mask_shape, device)
        #===================================将其变换到另一个空间======================================
        text_embeds = self.gelu(self.text_transform(text_embeds))
        
        #===================================get image emb======================================
        #######在扔到vit前先对维度转变到B*N,3,H,W上去,因为vit的输入是这样的########################
        if frame_input.dim() == 5:
            B, N, C, H, W = frame_input.shape
            frame_input = frame_input.view(B * N, C, H, W)
            output_shape = (B, N, -1)
        else:
            output_shape = (x.shape[0], -1)
        image_embeds = self.visual_backbone(frame_input)
        
        
        extend_image_masks = self.text_encoder.get_extended_attention_mask(frame_mask, frame_mask.size(), device)
        #============================维度变回来==================================
        image_embeds = image_embeds.view(*output_shape)
        #===================================将其变换到另一个空间======================================
        image_embeds = self.gelu(self.image_transform(image_embeds))
        #===================================时序编码器======================================
        #print(image_embeds.shape,frame_mask.shape)
        #image_embeds = self.visual_backbone_temporal_encoder(image_embeds,frame_mask)
        
        
        #===================在进入cross attention前加入tokentype embedding=========
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(title_mask)),
            image_embeds+ self.token_type_embeddings(torch.full_like(frame_mask, 1)),
        )
        
        #x, y = text_embeds, image_embeds
        # for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
        #     x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
        #     y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
        #     x, y = x1[0], y1[0]
        for image_layer in  self.cross_modal_image_layers:
            #x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            image_embeds = image_layer(image_embeds, text_embeds, extend_image_masks, extend_text_masks)[0]
            #x, y = x1[0], y1[0]
        #text_feats, image_feats = x, y
        #print(image_feats.shape,text_feats.shape)
        
        #cls_feats_text = self.cross_modal_text_pooler(x)
        #cls_feats_image = self.cross_modal_image_pooler(y)
        #cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        
        
        #text_feats = (text_feats * title_mask.unsqueeze(-1)).sum(1) / title_mask.sum(1).unsqueeze(-1)
        image_feats = (image_embeds * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1).unsqueeze(-1)
        #cls_feats = torch.cat([text_feats, image_embeds], dim=-1)
        
        #prediction = self.classifier(self.dropout(image_feats))
        prediction = self.classifier(self.dropout(image_feats))

        if inference:  # 预测
            return torch.argmax(prediction, dim=1)
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