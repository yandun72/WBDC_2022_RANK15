'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from xbert import BertConfig, BertForMaskedLM,BertModel
import torch
from torch import nn
from category_id_map import CATEGORY_ID_LIST
import torch.nn.functional as F
from config import parse_args
import numpy as np
args = parse_args()
from vison_model.swin import *
from vison_model.efficentformer import *

from vison_model.swinv2 import *
from vison_model.levit import LeViT_256,LeViT_192,LeViT_384
from transformers import VanModel, VanConfig
from transformers import BertModel as BertModel_tf
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0

class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None):
        super().__init__()

        embed_dim = 768
        # text encoder
        bert_config = BertConfig.from_json_file('./config.json')  # 'config_bert.json'
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config)
        
        
        #self.position_embeddings = nn.Embedding(33, embed_dim)
        
        # vision encoder
        vision_width = 768
        #
        #*****************************视觉模型*****************************
        #
        global args
        self.args = args
        self.vision_encoder = args.vision_model
        if self.vision_encoder == 'swin_tiny':
            self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        elif self.vision_encoder == 'swin_base':
            self.visual_backbone = swin_base(args.swin_pretrained_path)
        elif self.vision_encoder == 'swin_v2':
                self.visual_backbone = swinv2_base_patch4_window12_192_22k(num_classes=0)
                checkpoint = torch.load(args.swin_pretrained_path, map_location='cpu')['model']
                self.visual_backbone.load_state_dict(checkpoint, strict=False)
        if self.vision_encoder != 'swin_tiny':
            self.dim_to_768 = nn.Linear(1024,768)
            
        
        ##################################################################################
        self.clip = args.clip
        if args.clip == True:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.visual_projection = nn.Linear(768, 768, bias=False)
            self.text_projection = nn.Linear(768, 768, bias=False)
        
        self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))
        self.vision_cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.gelu = nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, inputs, inference=False):
        bs = inputs['frame_input'].shape[0]
        frames = inputs['frame_input'].shape[1]
        device = 'cuda'
        inputs['title_input'] = inputs['title_input'].to(device)
        inputs['title_mask'] = inputs['title_mask'].to(device)
        inputs['frame_input'] = inputs['frame_input'].to(device)
        inputs['frame_mask'] = inputs['frame_mask'].to(device)
        if inference == False:
            inputs['label'] = inputs['label'].to(device)
        

        # video encoder
        #vision_embeds = torch.zeros((bs,frames,1024),device=device)
        #for i in range(bs):
            #print(bs,self.visual_backbone((bs,frames,))
            #vision_embeds[i] = self.visual_backbone(inputs['frame_input'][i])
        vision_embeds = self.visual_backbone(inputs['frame_input'])
        if self.vision_encoder != 'swin_tiny':
            vision_embeds = self.dropout(self.dim_to_768(vision_embeds))
        
        
        video_features = vision_embeds
        video_masks = inputs['frame_mask']
        # cat cls
        #print(vision_embeds.shape)
        #vision_cls_tokens = self.vision_cls_token.repeat(bs, 1, 1).cuda()
        #vision_cls_mask = torch.ones((bs, 1)).to(device)
        #video_features = torch.cat((vision_cls_tokens, vision_embeds), dim=1)
        #video_masks = torch.cat((vision_cls_mask, inputs['frame_mask']), dim=1)
        # todo
        # add posiontion info 
        #video_features = self.text_encoder.
        # text encoder
        text_embeds = self.text_encoder(inputs['title_input'], attention_mask=inputs['title_mask'],
                                             return_dict=True,
                                             mode='text').last_hidden_state
        
        if self.clip == True:
            # normalized features
            image_embeds_clip=video_features[:,0,:]
            image_embeds_clip = self.visual_projection(image_embeds_clip)
            
            text_embeds_clip = text_embeds[:,0,:]
            text_embeds_clip = self.text_projection(text_embeds_clip)
            
            image_embeds_clip = image_embeds_clip / image_embeds_clip.norm(p=2, dim=-1, keepdim=True)
            text_embeds_clip = text_embeds_clip / text_embeds_clip.norm(p=2, dim=-1, keepdim=True)
            
            #print(image_embeds_clip.shape,text_embeds_clip.shape)
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds_clip, image_embeds_clip.t()) * logit_scale
            logits_per_image = logits_per_text.T
            loss_clip = clip_loss(logits_per_text)
            #print(loss_clip)
            #c = d
        # get momentum features
        output1 = self.text_encoder(    encoder_embeds=text_embeds,
                                        attention_mask=inputs['title_mask'],
                                        encoder_hidden_states=video_features,
                                        encoder_attention_mask=video_masks,
                                        return_dict=True,
                                        mode='fusion',
                                        ).last_hidden_state

        # output2 = self.text_encoder(    encoder_embeds=video_features,
        #                                 attention_mask=video_masks,
        #                                 encoder_hidden_states=text_embeds,
        #                                 encoder_attention_mask=inputs['title_mask'],
        #                                 return_dict=True,
        #                                 mode='fusion',
        #                                 ).last_hidden_state
        #output = output.mean(dim=1)  # [bs,768]
        #mask2 = inputs['title_mask']
        #output = (output * mask2.unsqueeze(-1)).sum(1) / mask2.sum(1).unsqueeze(-1)
        #cls_info = encoder_outputs[:,0,:]
        #output = torch.cat((output1[:,0,:],output2[:,0,:]),dim=-1)
        prediction = self.classifier(self.dropout(output1[:,0,:]))

        if inference:  # 预测
            return torch.argmax(prediction, dim=1)
        else:
            loss, accuracy, pred_label_id, label = self.cal_loss(prediction, inputs['label'])
            if self.clip == True:
                loss += loss_clip/self.args.clip_weight
            return loss, accuracy, pred_label_id, label

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label