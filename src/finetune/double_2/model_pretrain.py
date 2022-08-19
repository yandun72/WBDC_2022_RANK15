import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertModel,BertForMaskedLM
from bert_model import BertCrossLayer
from objectives import init_weights
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform_1 = nn.Linear(768, 768*2)
        self.transform_1.apply(init_weights)
        self.transform_2 = nn.Linear(768*2, 768)
        self.transform_2.apply(init_weights)
        self.dropout = torch.nn.Dropout(0.1)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.transform_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.transform_2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x
class Clip_Vit_Cross_Model_Pretrain(nn.Module):
    def __init__(self, bert_dir, num_top_layer):
        super().__init__()

        bert_config = BertConfig.from_pretrained(bert_dir)
        self.text_encoder = BertForMaskedLM.from_pretrained(bert_dir)
        self.text_transform = MLP()
        self.video_transform = MLP()
        self.text_transform_2 = MLP()
        self.video_transform_2 = MLP()
        
        self.token_type_embeddings = nn.Embedding(2, 768)
        self.token_type_embeddings.apply(init_weights)

        # ==================cross encoder========================
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(num_top_layer)])
        self.cross_modal_image_layers.apply(init_weights)
        

        self.dropout = torch.nn.Dropout(0.1)
        self.gelu = nn.GELU()
        
        '''
        ALBEF的东西
        '''
        self.temp = nn.Parameter(torch.ones([]) * 0.07)   
        self.queue_size = 4000
        self.momentum = 0.995
        self.itm_head = nn.Linear(768, 2)     
        self.mlm_probability = 0.15
        # create momentum models
        self.text_transform_m = MLP()
        self.video_transform_m = MLP()
        self.text_transform_2_m = MLP()
        self.video_transform_2_m = MLP()
        
        self.text_encoder_m = BertForMaskedLM.from_pretrained(bert_dir)
        self.model_pairs = [[self.text_transform,self.text_transform_m],
                            [self.video_transform,self.video_transform_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_transform_2,self.text_transform_2_m],
                            [self.video_transform_2,self.video_transform_2_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(768, 10000))
        self.register_buffer("text_queue", torch.randn(768, 10000))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, title_input, title_mask, frame_input, frame_mask):
        device = 'cuda'
        alpha = 0.4
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        frame_input_temp = (frame_input * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1).unsqueeze(-1)
        video_embeds_cls = self.video_transform(frame_input_temp)# video_embeds_cls 是进cross前的视频表征，算itc的时候需要用 
        video_atts = frame_mask
        
        text_output = self.text_encoder(title_input, title_mask,output_hidden_states=True).hidden_states[-1]
        
        text_output_cls = (text_output * title_mask.unsqueeze(-1)).sum(1) / title_mask.sum(1).unsqueeze(-1)
        text_output_cls = self.text_transform(text_output_cls) #text_output_cls是进cross前的文本表征，算itc的时候需要用 
        
        extend_image_masks = self.text_encoder.get_extended_attention_mask(frame_mask, frame_mask.size(), device)
        extend_text_masks = self.text_encoder.get_extended_attention_mask(title_mask, title_mask.size(), device)
        
        # get momentum features
        with torch.no_grad():
            self._momentum_update() #把动量模型的权重更新一下
            frame_input_temp_m = (frame_input * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1).unsqueeze(-1)
            video_embeds_cls_m = self.video_transform_m(frame_input_temp_m) #video_embeds_cls_m：bs,768
            video_feat_all = torch.cat([video_embeds_cls_m.t(),self.image_queue.clone().detach()],dim=1)  #video_embeds_cls_m.t()：768,bs;self.image_queue:768,10000,cat后768,10000+bs    
                                                  
            text_output_m = self.text_encoder(title_input, title_mask,output_hidden_states=True).hidden_states[-1]
            text_output_cls_m = (text_output_m * title_mask.unsqueeze(-1)).sum(1) / title_mask.sum(1).unsqueeze(-1)
            text_output_cls_m = self.text_transform(text_output_cls_m)
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
        text_embeds, image_embeds = (
            text_output + self.token_type_embeddings(torch.zeros_like(title_mask)),
            frame_input + self.token_type_embeddings(torch.full_like(frame_mask, 1)),
        )
        text_embeds = self.text_transform_2(text_embeds) #text_embeds：bs,maxlen,768
        image_embeds = self.video_transform_2(image_embeds) #text_embeds：bs,frame,768
        output_pos = image_embeds
        for image_layer in self.cross_modal_image_layers:
            output_pos = image_layer(output_pos, text_embeds, extend_image_masks, extend_text_masks)[0]
        
        
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

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([title_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all   = torch.cat([image_atts_neg,frame_mask],dim=0)
        image_atts_all_extend_itc = self.text_encoder.get_extended_attention_mask(image_atts_all, image_atts_all.size(), device)
        text_atts_all_extend_itc = self.text_encoder.get_extended_attention_mask(text_atts_all, text_atts_all.size(), device)
        output_neg = image_embeds_all
        for image_layer in self.cross_modal_image_layers:
            output_neg = image_layer(output_neg, text_embeds_all, image_atts_all_extend_itc, text_atts_all_extend_itc)[0]
        
        output_pos = (output_pos * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1).unsqueeze(-1)
        output_neg = (output_neg * image_atts_all.unsqueeze(-1)).sum(1) / image_atts_all.sum(1).unsqueeze(-1)
        vl_embeddings = torch.cat([output_pos,output_neg],dim=0)#shape:bs*3,768
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
        mlm_output = self.text_encoder(input_ids, attention_mask = title_mask,labels = labels)
        prediction_scores = mlm_output[1]
        masked_lm_loss =  mlm_output[0]
        soft_labels = F.softmax(logits_m[1],dim=-1) #动量模型预测的label，只是softmax软标签形式
        loss_distill = -torch.sum(F.log_softmax(prediction_scores, dim=-1)*soft_labels,dim=-1) #由于albef的BertForMaskedLM封装了计算soft label带来的损失，huggingface上没有，因此得手写一下这部分
        loss_distill = loss_distill[labels!=-100].mean()#prediction_scores是正常的模型的输出，soft_labels是动量模型的输出
        
        masked_lm_loss = (1-alpha)*masked_lm_loss + alpha*loss_distill
        return masked_lm_loss, loss_ita, loss_itm 
    
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