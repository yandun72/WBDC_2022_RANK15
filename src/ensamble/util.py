import logging
import random
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW,get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup

from category_id_map import lv2id_to_lv1id


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def setup_logging_2(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f'{args.savedmodel_path}/{args.seeds}/train_seed_{args.seeds}.log',
                        filemode='w',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def setup_logging(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=args.savedmodel_path+'/'+args.log_name+'.log',
                        filemode='w',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model,per_data_loader_len):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters_name = []
    epsion = 1.0
    #===================bert的分层学习率=========================================
    
    for i in range(11,-1,-1):
        if i != 1:    #主要在于layer.1和layer.10 layer.11这两个，匹配的话，不加if这个判断会导致在i = 1的时候，将layer.10的参数也包含进来了
            tmp = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone' not in n and 'cross_modal' not in n and f'layer.{i}' in n],'weight_decay': args.weight_decay,'lr':args.learning_rate*epsion},

            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'visual_backbone' not in n and 'cross_modal' not in n and f'layer.{i}' in n], 'weight_decay': 0.0,'lr':args.learning_rate*epsion}]
            optimizer_grouped_parameters.extend(tmp)

            tmp =  [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone' not in n and 'cross_modal' not in n and f'layer.{i}' in n]

            optimizer_grouped_parameters_name.extend(tmp)    
            tmp2 = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'visual_backbone' not in n and 'cross_modal' not in n and f'layer.{i}' in n]
        else:
            tmp = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone' not in n and 'cross_modal' not in n and 'layer.1' in n and 'layer.10' not in n  and 'layer.11' not in n],'weight_decay': args.weight_decay,'lr':args.learning_rate*epsion},

            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'visual_backbone' not in n and 'cross_modal' not in n and 'layer.1' in n and 'layer.10'   not in n and 'layer.11' not in n], 'weight_decay': 0.0,'lr':args.learning_rate*epsion}]
            optimizer_grouped_parameters.extend(tmp)

            tmp =  [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone' not in n and 'cross_modal' not in n and 'layer.1' in n and 'layer.10'   not in n and 'layer.11' not in n]

            optimizer_grouped_parameters_name.extend(tmp)    
            tmp2 = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'visual_backbone' not in n and 'cross_modal' not in n and 'layer.1' in n and 'layer.10'   not in n and 'layer.11' not in n]
        if args.local_rank ==0:
            print('============================下面这些参数的学习率是:',args.learning_rate*epsion)
            print(tmp)
            print(tmp2)
        optimizer_grouped_parameters_name.extend(tmp2)    
        epsion *= 0.95
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''
    #===================bert的embedding分层学习率=========================================
    tmp = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'embeddings' in n],'weight_decay': args.weight_decay,'lr':args.learning_rate*epsion},
            
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'embeddings' in n], 'weight_decay': 0.0,'lr':args.learning_rate*epsion}]
    optimizer_grouped_parameters.extend(tmp)
    
    tmp = [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'embeddings' in n]
    optimizer_grouped_parameters_name.extend(tmp)         
    tmp2 =    [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'embeddings' in n]
    if args.local_rank ==0:
        print('============================下面这些参数的学习率是:',args.learning_rate*epsion)
        print(tmp)
        print(tmp2)
    optimizer_grouped_parameters_name.extend(tmp2)
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''
    #===================swin的分层学习率=========================================
    epsion = 1.0
    for i in range(3,-1,-1):
        tmp = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone' in n and f'layers.{i}' in n and 'patch_embed' not in n and 'visual_backbone.norm' not in n and 'cross_modal' not in n],'weight_decay': args.weight_decay,'lr':args.vison_learning_rate*epsion},
            
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'visual_backbone'  in n and f'layers.{i}' in n and 'patch_embed' not in n and 'visual_backbone.norm' not in n and 'cross_modal' not in n], 'weight_decay': 0.0,'lr':args.vison_learning_rate*epsion}]
        optimizer_grouped_parameters.extend(tmp)
        
        tmp = [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone' in n and f'layers.{i}' in n and 'patch_embed' not in n and 'visual_backbone.norm' not in n and 'cross_modal' not in n]
        
        optimizer_grouped_parameters_name.extend(tmp)
        tmp2 = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'visual_backbone'  in n and f'layers.{i}' in n and 'patch_embed' not in n and 'visual_backbone.norm' not in n and 'cross_modal' not in n]
        if args.local_rank ==0:
                print('============================下面这些参数的学习率是:',args.vison_learning_rate*epsion)
                print(tmp)
                print(tmp2)
        optimizer_grouped_parameters_name.extend(tmp2)
        
        epsion *= 0.92
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''
    #===================swin的patch_embed分层学习率=========================================
    tmp = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'patch_embed' in n and 'visual_backbone.norm' not in n and 'cross_modal' not in n],'weight_decay': args.weight_decay,'lr':args.vison_learning_rate*epsion},
            
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)  and 'patch_embed' in n and 'visual_backbone.norm' not in n and 'cross_modal' not in n], 'weight_decay': 0.0,'lr':args.vison_learning_rate*epsion}]
    optimizer_grouped_parameters.extend(tmp)
    
    tmp = [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'patch_embed' in n and 'visual_backbone.norm' not in n and 'cross_modal' not in n]

    optimizer_grouped_parameters_name.extend(tmp)        
    tmp2 = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay)  and 'patch_embed' in n and 'visual_backbone.norm' not in n and 'cross_modal' not in n]
    if args.local_rank ==0:
        print('============================下面这些参数的学习率是:',args.vison_learning_rate*epsion)
        print(tmp)
        print(tmp2)
    optimizer_grouped_parameters_name.extend(tmp2)
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''
    #===================swin的visual_backbone.norm分层学习率=========================================
    tmp = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone.norm'  in n and 'cross_modal' not in n],'weight_decay': args.weight_decay,'lr':args.vison_learning_rate},
            
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)   and 'visual_backbone.norm'  in n and 'cross_modal' not in n], 'weight_decay': 0.0,'lr':args.vison_learning_rate}]
    optimizer_grouped_parameters.extend(tmp)
    
    tmp = [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone.norm'  in n and 'cross_modal' not in n]

    optimizer_grouped_parameters_name.extend(tmp)        
    tmp2 =  [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay)   and 'visual_backbone.norm'  in n and 'cross_modal' not in n]
    if args.local_rank ==0:
        print('============================下面这些参数的学习率是:',args.vison_learning_rate)
        print(tmp)
        print(tmp2)
    optimizer_grouped_parameters_name.extend(tmp2)
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''
    #===================swin的cross_modal_image_layers分层学习率========================================= 
    epsion = 1.0
    for i in range(args.num_top_layer-1,-1,-1):
        tmp = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'cross_modal' in n and f'layers.{i}' in n],'weight_decay': args.weight_decay,'lr':args.vison_learning_rate*epsion},
            
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'cross_modal' in n and f'layers.{i}' in n], 'weight_decay': 0.0,'lr':args.vison_learning_rate*epsion}]
        optimizer_grouped_parameters.extend(tmp)
        
        tmp = [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'cross_modal' in n and f'layers.{i}' in n]
        
        optimizer_grouped_parameters_name.extend(tmp)    
        tmp2 =  [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'cross_modal' in n and f'layers.{i}' in n]
        if args.local_rank ==0:
                print('============================下面这些参数的学习率是:',args.vison_learning_rate*epsion)
                print(tmp)
                print(tmp2)
        optimizer_grouped_parameters_name.extend(tmp2)
        
        epsion *= 0.93
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''    
    #===================分类的学习率========================================= 
    tmp = [
        {'params': [p for n, p in model.named_parameters() if 'classifier.weight' in n],'weight_decay': args.weight_decay,'lr':args.classifier_learning_rate},
            
        {'params': [p for n, p in model.named_parameters() if 'classifier.bias' in n], 'weight_decay': 0.0,'lr':args.classifier_learning_rate}]
    optimizer_grouped_parameters.extend(tmp)
    
    tmp = [n for n, p in model.named_parameters() if 'classifier.weight' in n]

    optimizer_grouped_parameters_name.extend(tmp)        
    tmp2 =  [n for n, p in model.named_parameters() if 'classifier.bias' in n]
    if args.local_rank ==0:
        print('============================下面这些参数的学习率是:',args.classifier_learning_rate)
        print(tmp)
        print(tmp2)
    optimizer_grouped_parameters_name.extend(tmp2)
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''    
    
    params = list(model.named_parameters())
    all_params = [n for n,p in params]
    a = set(optimizer_grouped_parameters_name)
    c = []
    for x in a:
        if x not in c:
            c.append(x)
        else:
            print(x)
    
    b = set(all_params)
    if a== b:
        if args.local_rank ==0:
            print('=====================参数都囊括到我们的分组啦=====================')
    else:
        if args.local_rank ==0:
            print('=====================参数不在我们的分组有=====================')
            print(b - a)
            print(f'===================他们的学习率将设为默认的{args.vison_learning_rate}=========================')
    if args.local_rank ==0:
        print(f'==================total steps[{per_data_loader_len*args.max_epochs}]====================warm steps[{int(per_data_loader_len*args.max_epochs*0.15)}]==================')
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.vison_learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(per_data_loader_len*args.max_epochs*0.15),
                                                num_training_steps=per_data_loader_len*args.max_epochs)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results
