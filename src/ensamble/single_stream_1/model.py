import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertConfig
from category_id_map import CATEGORY_ID_LIST
from model_cfg import *
from pretrain_cfg import *
from qq_uni_model import *



class VLModel(nn.Module):
    def __init__(self, args,is_test=False):
        super().__init__()
        if is_test != True:
            pretrain_state = torch.load('../../pretrain/single_1/checkpoint/model_epoch_9.pth, map_location='cpu')
            pretrain_model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=args.bert_dir, task=PRETRAIN_TASK)
            pretrain_model_m = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=args.bert_dir, task=PRETRAIN_TASK)
            pretrain_model.load_state_dict(pretrain_state,strict=False)
        else:
            pretrain_model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=args.bert_dir, task=PRETRAIN_TASK) 
            #pretrain_model_m = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=args.bert_dir, task=PRETRAIN_TASK) 
        self.bert = pretrain_model.roberta.bert  
        # create momentum models
        #self.bert_m = pretrain_model_m.roberta.bert
        #self.model_pairs = [[self.bert,self.bert_m]]
        
        self.args = args
        

        self.classifier = nn.Linear(768*4, len(CATEGORY_ID_LIST))
        #self.classifier_m = nn.Linear(768*4, len(CATEGORY_ID_LIST))
        self.dropout = torch.nn.Dropout(0.1)
        self.momentum = 0.995 # copy as albef
        
    def forward(self, inputs, inference=False):
        bs = inputs['frame_input'].shape[0]
        frames = inputs['frame_input'].shape[1]
        
        device = 'cuda'
        title_input = inputs['title_input'].to(device)
        title_mask = inputs['title_mask'].to(device)
        frame_input = inputs['frame_input'].to(device)
        frame_mask = inputs['frame_mask'].to(device)
        text_token_type = inputs['text_token_type'].to(device)
        video_token_type = inputs['video_token_type'].to(device)
        
        if inference == False:
            label = inputs['label'].to(device)
        
        #inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        encoder_outputs,text_embedding,video_embedding = self.bert(frame_input,frame_mask,title_input,title_mask,text_token_type,video_token_type)
        cls_info = encoder_outputs[:,0,:]
        mask2 = torch.cat([frame_mask, title_mask], 1)
        text_video_outputs = (encoder_outputs * mask2.unsqueeze(-1)).sum(1) / mask2.sum(1).unsqueeze(-1)
        final_feature = torch.cat((cls_info,text_embedding,video_embedding,text_video_outputs),dim=-1)
        prediction = self.classifier(self.dropout(final_feature))
        
        # # get momentum features
        # with torch.no_grad():
        #     self._momentum_update()
        #     encoder_outputs_m,text_embedding_m,video_embedding_m = self.bert(frame_input,frame_mask,title_input,title_mask,text_token_type,video_token_type)
        #     cls_info_m = encoder_outputs_m[:,0,:]
        #     mask2 = torch.cat([frame_mask, title_mask], 1)
        #     text_video_outputs_m = (encoder_outputs_m * mask2.unsqueeze(-1)).sum(1) / mask2.sum(1).unsqueeze(-1)
        #     final_feature_m = torch.cat((cls_info_m,text_embedding_m,video_embedding_m,text_video_outputs_m),dim=-1)
        #     prediction_m = self.classifier_m(self.dropout(final_feature_m))
        
        if inference:  # 预测
            return torch.argmax(prediction, dim=1),prediction
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction,label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        # loss_m = F.cross_entropy(prediction_m, label)
        # loss = 0.9*loss + 0.1*loss_m
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    #copy from https://github.com/salesforce/ALBEF/blob/main/models/model_pretrain.py
    #使用albef的动量蒸馏
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
