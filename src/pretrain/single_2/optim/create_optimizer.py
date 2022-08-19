from transformers import AdamW
from transformers import AdamW,get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
from config.pretrain_cfg import *

def create_optimizer(model, model_lr={'others':5e-3, 'nextvlad':5e-4, 'roberta':5e-5},
                     weight_decay=0.01, layerwise_learning_rate_decay=0.975,
                     adam_epsilon=1e-6, use_bertadam = False):
    # Set learning_rates for each layers
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for layer_name in model_lr:
        lr = model_lr[layer_name]
#         #Layer decay learning rate
#         if layer_name == 'roberta':  # Robert 使用 layerwise_decay
#             layers =  [getattr(model, layer_name).embeddings] + list(getattr(model, layer_name).encoder.layer)
#             layers.reverse()
#             for layer in layers:
#                 lr *= layerwise_learning_rate_decay
#                 optimizer_grouped_parameters += [
#                     {
#                         "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
#                         "weight_decay": weight_decay,
#                         "lr": lr,
#                     },
#                     {
#                         "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
#                         "weight_decay": 0.0,
#                         "lr": lr,
#                     },
#                 ]
        if layer_name != 'others':  # 设定了特定 lr 的 layer
             optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                          and layer_name in n)],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                          and layer_name in n)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
             ]
        else:  # 其他，默认学习率
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                          and not any(name in n for name in model_lr))],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                          and not any(name in n for name in model_lr))],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=model_lr['roberta'],
        eps=adam_epsilon,
        correct_bias=not use_bertadam
    )
    return optimizer

def build_optimizer(args, model,per_data_loader_len,logging):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters_name = []
    epsion = 1.0
    #===================bert的分层学习率=========================================
    
    for i in range(args.text_transforemr_layers-1,-1,-1):
        if i != 1:    #主要在于layer.1和layer.10 layer.11这两个，匹配的话，不加if这个判断会导致在i = 1的时候，将layer.10的参数也包含进来了
            tmp = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and  f'layer.{i}' in n],'weight_decay': args.weight_decay,'lr':args.learning_rate*epsion},

            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and f'layer.{i}' in n], 'weight_decay': 0.0,'lr':args.learning_rate*epsion}]
            optimizer_grouped_parameters.extend(tmp)

            tmp =  [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and  f'layer.{i}' in n]

            optimizer_grouped_parameters_name.extend(tmp)    
            tmp2 = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and  f'layer.{i}' in n]
        else:
            tmp = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'layer.1' in n and 'layer.10' not in n  and 'layer.11' not in n],'weight_decay': args.weight_decay,'lr':args.learning_rate*epsion},

            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'layer.1' in n and 'layer.10'  not in n and 'layer.11' not in n], 'weight_decay': 0.0,'lr':args.learning_rate*epsion}]
            optimizer_grouped_parameters.extend(tmp)

            tmp =  [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'layer.1' in n and 'layer.10'   not in n and 'layer.11' not in n]

            optimizer_grouped_parameters_name.extend(tmp)    
            tmp2 = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'layer.1' in n and 'layer.10'   not in n and 'layer.11' not in n]
        if args.local_rank ==0:
            print(f'============================bert layer:{i}下面这些参数的学习率是:',args.learning_rate*epsion)
            print(tmp)
            print(tmp2)
            logging.info(f'============================bert layer:{i}下面这些参数的学习率是:{args.learning_rate*epsion}')
            logging.info(str(tmp))
            logging.info(str(tmp2))
        optimizer_grouped_parameters_name.extend(tmp2)    
        epsion *= 0.95
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''
    logging.info('''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    ''')
    #===================bert的embedding分层学习率=========================================
    tmp = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'embeddings' in n],'weight_decay': args.weight_decay,'lr':args.learning_rate*epsion},
            
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'embeddings' in n], 'weight_decay': 0.0,'lr':args.learning_rate*epsion}]
    optimizer_grouped_parameters.extend(tmp)
    
    tmp = [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'embeddings' in n]
    optimizer_grouped_parameters_name.extend(tmp)         
    tmp2 =    [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'embeddings' in n]
    if args.local_rank ==0:
        print(f'============================bert embedding下面这些参数的学习率是:',args.learning_rate*epsion)
        print(tmp)
        print(tmp2)
        logging.info(f'============================bert embedding下面这些参数的学习率是:{args.learning_rate*epsion}')
        logging.info(str(tmp))
        logging.info(str(tmp2))
    optimizer_grouped_parameters_name.extend(tmp2)
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''
    logging.info('''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    ''')
    
    #===================itm 和 mfm mlm itc分类的学习率========================================= 
    tmp = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and ('itm' in n or 'cls' in n or 'mvm' in n or 'itc' in n)],'weight_decay': args.weight_decay,'lr':args.classifier_learning_rate},
            
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and ('itm' in n or 'cls' in n or 'mvm' in n or 'itc' in n)], 'weight_decay': 0.0,'lr':args.classifier_learning_rate}]
    optimizer_grouped_parameters.extend(tmp)
    
    tmp = [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and ('itm' in n or 'cls' in n or 'mvm' in n or 'itc' in n)]
    optimizer_grouped_parameters_name.extend(tmp)        
    tmp2 =  [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and ('itm' in n or 'cls' in n or 'mvm' in n or 'itc' in n)]
    
    if args.local_rank ==0:
            print(f'============================itm mfm下面这些参数的学习率是:',args.classifier_learning_rate)
            print(tmp)
            print(tmp2)
            logging.info(f'============================itm mfm下面这些参数的学习率是:{args.classifier_learning_rate}')
            logging.info(str(tmp))
            logging.info(str(tmp2))
    optimizer_grouped_parameters_name.extend(tmp2)
    
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
    if a == b:
        if args.local_rank ==0:
            print('=====================参数都囊括到我们的分组啦=====================')
            logging.info('=====================参数都囊括到我们的分组啦=====================')
    else:
        if args.local_rank ==0:
            print('=====================参数不在我们的分组有=====================')
            logging.info('=====================参数不在我们的分组有=====================')
            print(b - a)
            logging.info(str(b-a))
            print(f'===================他们的学习率将设为默认的{args.classifier_learning_rate}=========================')
            logging.info(f'===================他们的学习率将设为默认的{args.classifier_learning_rate}=========================')
    if args.local_rank ==0:
        print(f'==================total steps[{per_data_loader_len*NUM_EPOCHS}]====================warm steps[{int(per_data_loader_len*NUM_EPOCHS*WARMUP_RATIO)}]==================')
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(per_data_loader_len*NUM_EPOCHS*WARMUP_RATIO),
                                                num_training_steps=per_data_loader_len*NUM_EPOCHS)
    return optimizer, scheduler


def build_optimizer_continue(args, model,per_data_loader_len):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters_name = []
    epsion = 1.0
    #===================bert的分层学习率=========================================
    
    for i in range(args.text_transforemr_layers-1,-1,-1):
        if i != 1:    #主要在于layer.1和layer.10 layer.11这两个，匹配的话，不加if这个判断会导致在i = 1的时候，将layer.10的参数也包含进来了
            tmp = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and  f'layer.{i}' in n],'weight_decay': args.weight_decay,'lr':args.learning_rate*epsion},

            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and f'layer.{i}' in n], 'weight_decay': 0.0,'lr':args.learning_rate*epsion}]
            optimizer_grouped_parameters.extend(tmp)

            tmp =  [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and  f'layer.{i}' in n]

            optimizer_grouped_parameters_name.extend(tmp)    
            tmp2 = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and  f'layer.{i}' in n]
        else:
            tmp = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'layer.1' in n and 'layer.10' not in n  and 'layer.11' not in n],'weight_decay': args.weight_decay,'lr':args.learning_rate*epsion},

            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'layer.1' in n and 'layer.10'  not in n and 'layer.11' not in n], 'weight_decay': 0.0,'lr':args.learning_rate*epsion}]
            optimizer_grouped_parameters.extend(tmp)

            tmp =  [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'layer.1' in n and 'layer.10'   not in n and 'layer.11' not in n]

            optimizer_grouped_parameters_name.extend(tmp)    
            tmp2 = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'layer.1' in n and 'layer.10'   not in n and 'layer.11' not in n]
        if args.local_rank ==0:
            print(f'============================bert layer:{i}下面这些参数的学习率是:',args.learning_rate*epsion)
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
        print(f'============================bert embedding下面这些参数的学习率是:',args.learning_rate*epsion)
        print(tmp)
        print(tmp2)
        
    optimizer_grouped_parameters_name.extend(tmp2)
    '''
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    *************************************************************************************************************
    '''
    
    
    #===================itm 和 mfm mlm itc分类的学习率========================================= 
    tmp = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and ('itm' in n or 'lm' in n or 'mvm' in n or 'itc' in n)],'weight_decay': args.weight_decay,'lr':args.classifier_learning_rate},
            
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and ('itm' in n or 'lm' in n or 'mvm' in n or 'itc' in n)], 'weight_decay': 0.0,'lr':args.classifier_learning_rate}]
    optimizer_grouped_parameters.extend(tmp)
    
    tmp = [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and ('itm' in n or 'lm' in n or 'mvm' in n or 'itc' in n)]
    optimizer_grouped_parameters_name.extend(tmp)        
    tmp2 =  [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and ('itm' in n or 'lm' in n or 'mvm' in n or 'itc' in n)]
    
    if args.local_rank ==0:
            print(f'============================itm mfm下面这些参数的学习率是:',args.classifier_learning_rate)
            print(tmp)
            print(tmp2)
            
    optimizer_grouped_parameters_name.extend(tmp2)
    
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
    if a == b:
        if args.local_rank ==0:
            print('=====================参数都囊括到我们的分组啦=====================')
            
    else:
        if args.local_rank ==0:
            print('=====================参数不在我们的分组有=====================')
            #logging.info('=====================参数不在我们的分组有=====================')
            print(b - a)
            #logging.info(str(b-a))
            print(f'===================他们的学习率将设为默认的{args.learning_rate}=========================')
            #logging.info(f'===================他们的学习率将设为默认的{args.learning_rate}=========================')
    if args.local_rank ==0:
        print(f'==================total steps[{per_data_loader_len*NUM_EPOCHS}]====================warm steps[{int(per_data_loader_len*NUM_EPOCHS*WARMUP_RATIO)}]==================')
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(per_data_loader_len*NUM_EPOCHS*WARMUP_RATIO),
                                                num_training_steps=per_data_loader_len*NUM_EPOCHS)
    return optimizer, scheduler