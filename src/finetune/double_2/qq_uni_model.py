import math
import torch.nn.functional as F
from masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from model_pretrain import *
from pretrain_cfg import *

class QQUniModel(nn.Module):
    def __init__(self, args, task=['itm', 'mfm','mlm']):
        super().__init__()
        # 加载bert_config模型
        uni_bert_cfg = BertConfig.from_pretrained(f'{args.bert_dir}/config.json')
        self.task = task
        
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
            #self.num_class = cfg['NUM_CLASSES']
            self.vocab_size = uni_bert_cfg.vocab_size
            
        if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg)

        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1)

        self.roberta = UniBertForMaskedLM(uni_bert_cfg)

    def forward(self, video_feature, video_mask, text_input_ids, text_mask,task=['itm', 'mfm','mlm']):

        loss, pred = 0, None
        b = video_feature.shape[0]
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
        
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device) # [SEP] 卡 MASK 大师 [SEP]
            return_mlm = True
            
        if 'mfm' in sample_task:
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)

        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)

        # concat features
        # 回传特征
        video_features,image_cls,text_feature = self.roberta(video_feature, video_mask, text_input_ids, text_mask)
        
        if 'mlm' in sample_task:
            pred = text_feature.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss/2
            
        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(video_features)
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input,
                                                     video_mask, video_label, normalize=False)
            loss += masked_vm_loss/3
            
        if 'itm' in sample_task:
            pred = self.newfc_itm(image_cls)
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss*2
            
        return (pred, loss, itm_loss, masked_vm_loss,masked_lm_loss)


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
        masked_logits = logits_matrix + (1. - mask_matrix) * -100

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mfm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.mfm_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.mfm_dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.mfm_LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.mfm_decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.mfm_bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.mfm_decoder.bias = self.mfm_bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.mfm_decoder(hidden_states)
        return hidden_states


class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask):
        vision_embeds,image_cls,text_embeds = self.bert(text_input_ids, text_mask, video_feature, video_mask)

        return vision_embeds,image_cls,self.cls(text_embeds)


class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.meter = Clip_Vit_Cross_Model(BERT_PATH, NUM_TOP_LAYER)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, text_input_ids, text_mask, video_feature, video_mask):
        vision_embeds,image_cls,text_embeds = self.meter(text_input_ids, text_mask, video_feature, video_mask)

        return vision_embeds,image_cls,text_embeds