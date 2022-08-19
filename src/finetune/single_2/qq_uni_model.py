#%%writefile qqmodel/qq_uni_model.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from transformers import BertForMaskedLM
from pretrain_cfg import *
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform_1 = nn.Linear(768, 768)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.transform_1(x)
        x = self.gelu(x)
        return x
class Position_Embedding_For_Video(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embeddings = nn.Embedding(16, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("position_ids", torch.arange(16).expand((1, -1)))
    def forward(self, embeddings):
        position_ids = self.position_ids[:, 0 : 16]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class Single_Stream_Model_Pretrain(nn.Module):
        def __init__(self,bert_dir):
            super().__init__()
            bert_config = BertConfig.from_pretrained(bert_dir)
            self.text_encoder = BertForMaskedLM.from_pretrained(bert_dir)
            self.video_position_embeddings = Position_Embedding_For_Video()

            self.text_transform_2 = MLP()
            self.video_transform_2 = MLP()
            self.gelu = nn.GELU()
            #mfm用的
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(bert_config)
            '''
            ALBEF的东西
            '''
            self.temp = nn.Parameter(torch.ones([]) * 0.07)   
            self.queue_size = 8800
            self.momentum = 0.995
            self.itm_head = nn.Linear(768*3, 2)     
            self.mlm_probability = 0.15
            # create momentum models
            self.text_transform_2_m = MLP()
            self.video_transform_2_m = MLP()

            self.text_encoder_m = BertForMaskedLM.from_pretrained(bert_dir)
            self.model_pairs = [[self.text_encoder,self.text_encoder_m],
                                [self.text_transform_2,self.text_transform_2_m],
                                [self.video_transform_2,self.video_transform_2_m],
                               ]

            self.copy_params()

            # create the queue
            self.register_buffer("image_queue", torch.randn(768, self.queue_size))
            self.register_buffer("text_queue", torch.randn(768, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        def forward(self, frame_input,frame_mask,title_input, title_mask, text_tokne_type,video_token_type):
            device = 'cuda'
            alpha = 0.4
            with torch.no_grad():
                self.temp.clamp_(0.001,0.5)
            frame_input = self.video_position_embeddings(frame_input) #为视频的帧添加位置信息
            frame_input_normal = self.video_transform_2(frame_input)
            video_embeds_cls = (frame_input_normal * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1).unsqueeze(-1)
            video_atts = frame_mask

            text_output = self.text_encoder.bert.embeddings(input_ids=title_input) #只用embedding信息做itc的loss计算
            text_output_cls = (text_output * title_mask.unsqueeze(-1)).sum(1) / title_mask.sum(1).unsqueeze(-1)
            #text_output_cls = self.text_transform(text_output_cls) #text_output_cls是进cross前的文本表征，算itc的时候需要用 

            extend_image_masks = self.text_encoder.get_extended_attention_mask(frame_mask, frame_mask.size(), device)
            extend_text_masks = self.text_encoder.get_extended_attention_mask(title_mask, title_mask.size(), device)

            # get momentum features
            with torch.no_grad():
                self._momentum_update() #把动量模型的权重更新一下
                #frame_input_temp_m = (frame_input * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1).unsqueeze(-1)
                video_embeds_cls_m = self.video_transform_2_m(frame_input) #video_embeds_cls_m：bs,768
                video_embeds_cls_m = (video_embeds_cls_m * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1).unsqueeze(-1)
                video_feat_all = torch.cat([video_embeds_cls_m.t(),self.image_queue.clone().detach()],dim=1)  #video_embeds_cls_m.t()：768,bs;self.image_queue:768,10000,cat后768,10000+bs    

                text_output_m = self.text_encoder_m.bert.embeddings(input_ids=title_input) #只用embedding信息做itc的loss计算
                text_output_cls_m = (text_output_m * title_mask.unsqueeze(-1)).sum(1) / title_mask.sum(1).unsqueeze(-1)
                #text_output_cls_m = self.text_transform(text_output_cls_m)
                text_feat_all = torch.cat([text_output_cls_m.t(),self.text_queue.clone().detach()],dim=1)


                sim_i2t_m = video_embeds_cls_m @ text_feat_all / self.temp #相似度直接通过矩阵相乘来代替
                sim_t2i_m = text_output_cls_m @ video_feat_all / self.temp     

                sim_targets = torch.zeros(sim_i2t_m.size()).to(title_input.device)
                sim_targets.fill_diagonal_(1)#将对角线全部设为1          

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

            sim_i2t = video_embeds_cls @ text_feat_all / self.temp 
            sim_t2i = text_output_cls  @ video_feat_all / self.temp 

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

            loss_ita = (loss_i2t+loss_t2i)/2

            self._dequeue_and_enqueue(video_embeds_cls_m, text_output_cls_m)

            ###=================================###
            # forward the positve image-text pair
            text_embeds = text_output #text_embeds：bs,maxlen,768
            image_embeds = frame_input_normal #text_embeds：bs,frame,768
            embedding_output = torch.cat([image_embeds, text_embeds], 1)

            mask = torch.cat([frame_mask, title_mask], 1)
            mask2 = mask.clone()
            video_mask = frame_mask.clone()
            mask = mask[:, None, None, :]
            mask = (1.0 - mask) * -10000.0

            encoder_outputs = self.text_encoder.bert.encoder(embedding_output, attention_mask=mask)[0]
            
            B,M,D = frame_input.shape
            video_embedding = (encoder_outputs[:,0:M,:] * video_mask.unsqueeze(-1)).sum(1) / video_mask.sum(1).unsqueeze(-1)
            text_embedding = (encoder_outputs[:,M:,:] * title_mask.unsqueeze(-1)).sum(1) / title_mask.sum(1).unsqueeze(-1)
            #cls_info = encoder_outputs[:,0,:]
            text_video_outputs = (encoder_outputs * mask2.unsqueeze(-1)).sum(1) / mask2.sum(1).unsqueeze(-1)
            output_pos = torch.cat((text_embedding,video_embedding,text_video_outputs),dim=-1)

            with torch.no_grad():
                bs = text_embeds.size(0)          
                weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1) #weights_i2t是sim_i2t的权重
                weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)

                weights_i2t.fill_diagonal_(0)#将对角线设为0
                weights_t2i.fill_diagonal_(0)

            # select a negative image for each text
            image_embeds_neg = [] 
            image_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
                image_atts_neg.append(frame_mask[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0) #image_embeds_neg顺序被打乱   
            image_atts_neg = torch.stack(image_atts_neg,dim=0)    

            # select a negative text for each image
            text_embeds_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_embeds_neg.append(text_embeds[neg_idx])
                text_atts_neg.append(title_mask[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   #text_embeds_neg顺序被打乱   
            text_atts_neg = torch.stack(text_atts_neg,dim=0)    
            
            #先横向按照第一维度的拼接两个，再将这两个按照第0维的纵向拼接；这点和双流区别很大。
            neg_1 = torch.cat((image_embeds_neg,text_embeds),dim=1)
            neg_1_attn = torch.cat((image_atts_neg,title_mask),dim=1)
            neg_2 = torch.cat((image_embeds,text_embeds_neg),dim=1)
            neg_2_attn = torch.cat((frame_mask,text_atts_neg),dim=1)
            video_mask_neg = torch.cat((image_atts_neg,frame_mask),dim=0)
            text_mask_neg =  torch.cat((title_mask,text_atts_neg),dim=0)
            embedding_neg_all = torch.cat([neg_1, neg_2],0)
            neg_attn_all = torch.cat([neg_1_attn, neg_2_attn],0)
            neg_attn_all_2 = neg_attn_all.clone()
            neg_attn_all = neg_attn_all[:, None, None, :]
            neg_attn_all = (1.0 - neg_attn_all) * -10000.0
            
            encoder_outputs_neg = self.text_encoder.bert.encoder(embedding_neg_all, attention_mask=neg_attn_all)[0]
            B,M,D = frame_input.shape
            video_embedding_neg = (encoder_outputs_neg[:,0:M,:] * video_mask_neg.unsqueeze(-1)).sum(1) / video_mask_neg.sum(1).unsqueeze(-1)
            text_embedding_neg = (encoder_outputs_neg[:,M:,:] * text_mask_neg.unsqueeze(-1)).sum(1) / text_mask_neg.sum(1).unsqueeze(-1)
            text_video_outputs_neg = (encoder_outputs_neg * neg_attn_all_2.unsqueeze(-1)).sum(1) / neg_attn_all_2.sum(1).unsqueeze(-1)
            output_neg =  torch.cat((text_embedding_neg,video_embedding_neg,text_video_outputs_neg),dim=-1)

            vl_embeddings = torch.cat([output_pos,output_neg],dim=0)#shape:bs*3,768*3
            vl_output = self.itm_head(vl_embeddings)#二分类，让模型判断是不是匹配的video-text对            

            itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                                   dim=0).to(output_pos.device) #itm的label，前bs个video-text对的label是1，后2bs个是0
            loss_itm = F.cross_entropy(vl_output, itm_labels)

            ##================= MLM ========================##                
            input_ids = title_input.clone()
            labels = title_input.clone()

            probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
            input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, input_ids.device, targets=labels,
                                          probability_matrix = probability_matrix) 

            with torch.no_grad():
                logits_m = self.text_encoder_m(input_ids,attention_mask = title_mask,labels = labels)    
            mlm_output = self.text_encoder(input_ids,attention_mask = title_mask,labels = labels)
            prediction_scores = mlm_output[1]
            masked_lm_loss =  mlm_output[0]
            soft_labels = F.softmax(logits_m[1],dim=-1) #动量模型预测的label，只是softmax软标签形式
            loss_distill = -torch.sum(F.log_softmax(prediction_scores, dim=-1)*soft_labels,dim=-1) #由于albef的BertForMaskedLM封装了计算soft label带来的损失，huggingface上没有，因此得手写一下这部分
            loss_distill = loss_distill[labels!=-100].mean()#prediction_scores是正常的模型的输出，soft_labels是动量模型的输出

            masked_lm_loss = (1-alpha)*masked_lm_loss + alpha*loss_distill
            
            ##================= MfM ========================##    
            vm_input = image_embeds
            input_feature, video_label = self.vm.torch_mask_frames(image_embeds.cpu(), frame_mask.cpu())
            video_feature = input_feature.to(image_embeds.device)
            video_label = video_label.to(frame_input.device)
            
            # forward the positve image-text pair
            text_embeds_mfm = text_output #text_embeds：bs,maxlen,768
            #此时有些video_feature中的帧已经被mask了
            image_embeds_mfm = image_embeds #text_embeds：bs,frame,768
            embedding_output_mfm = torch.cat([image_embeds_mfm, text_embeds_mfm], 1)
            encoder_outputs_mfm = self.text_encoder.bert.encoder(embedding_output_mfm, attention_mask=mask)[0]
            B,M,D = frame_input.shape
            video_embedding_mfm = encoder_outputs_mfm[:,0:M,:] 
            
            
            vm_output = self.roberta_mvm_lm_header(video_embedding_mfm)
            #print(vm_output.shape,vm_input.shape,video_mask.shape,video_label.shape)
            masked_mfm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     video_mask, video_label, normalize=False)
            
            return masked_lm_loss, loss_ita, loss_itm,masked_mfm_loss 
        
        
        def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
            if normalize:
                video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
                video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

            afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

            video_tr = video_feature_input.permute(2, 0, 1)
            video_tr = video_tr.view(video_tr.shape[0], -1)

            logits_matrix = torch.mm(afm_scores_tr, video_tr)
            if normalize:
                logits_matrix = logits_matrix / temp

            video_mask_float = video_mask.to(dtype=torch.float)
            mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
            masked_logits = logits_matrix + (1. - mask_matrix) * -100.0

            logpt = F.log_softmax(masked_logits, dim=-1)
            logpt = torch.diag(logpt)
            nce_loss = -logpt

            video_labels_index_mask = (video_labels_index != -100)
            nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
            nce_loss = nce_loss.mean()
            return nce_loss
        #from albef，copy_params复制初始权重给动量模型的参数
        @torch.no_grad()    
        def copy_params(self):
            for model_pair in self.model_pairs:           
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data.copy_(param.data)  # initialize
                    param_m.requires_grad = False  # not update by gradient    

                
        @torch.no_grad() #动量模型的权重的更新不是通过训练，而是正常模型param的指数移动平均  
        def _momentum_update(self):
            for model_pair in self.model_pairs:           
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()): #遍历每一组的所有的参数
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
        
        @torch.no_grad()
        def _dequeue_and_enqueue(self, image_feat, text_feat):
            # gather keys before updating queue
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)

            batch_size = image_feats.shape[0]

            ptr = int(self.queue_ptr)
            assert self.queue_size % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T #替换掉一个batch内的内容
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            ptr = (ptr + batch_size) % self.queue_size  # move pointer #指针循环移动

            self.queue_ptr[0] = ptr       
            
        def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
            if masked_indices is None:                                       
                masked_indices = torch.bernoulli(probability_matrix).bool()
            pad_token_id = 0   
            cls_token_id = 101
            mask_token_id = 103
            masked_indices[input_ids == pad_token_id] = False
            masked_indices[input_ids == cls_token_id] = False
            
            if targets is not None:
                targets[~masked_indices] = -100 # We only compute loss on masked tokens            

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = mask_token_id

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
            input_ids[indices_random] = random_words[indices_random]                     
            # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
            
            if targets is not None:
                return input_ids, targets
            else:
                return input_ids
@torch.no_grad()
def concat_all_gather(tensor):#把多卡的tensor聚合起来，后面一起替换掉队列当中的内容
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
            

class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
      
        