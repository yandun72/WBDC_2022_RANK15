import argparse

config = {
    'log_name': 'macbert',
    "seed": 2012,
    "dropout": 0.25,
    "train_annotation": '/home/tione/notebook/data/annotations/labeled.json',
    #"test_annotation": "/home/tione/notebook/data/annotations/labeled.json",
    "test_annotation": '/opt/ml/input/data/annotations/test.json',
    "train_zip_feats": '/home/tione/notebook/data/Vit_B_16_unlabeled_zip_feats',
    "test_zip_feats": '/opt/ml/input/data/zip_frames/test',
    #"test_zip_feats": '/home/tione/notebook/data/zip_frames/labeled',
    "test_output_csv": 'data/base_result.csv',

    "val_ratio": 0.1,  # 验证集大小
    "batch_size": 32,
    "accumlate_step":1,
    "val_batch_size": 50,
    "test_batch_size": 125,
    "prefetch": 16,
    "num_workers": 8,
    "savedmodel_path": './save',
    "ckpt_file": './save/v1/model_.bin',

    "best_score": 0.5,
    'max_epochs': 5,
    "max_steps": 14000,
    "print_steps": 100,
    "warmup_steps": 1000,
    "minimum_lr": 0.,
    "learning_rate": 1e-4,
    "vison_learning_rate": 2e-4,
    "classifier_learning_rate": 4e-5,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-6,


    "bert_dir": '../../../opensource_models/macbert',
    # "bert_dir" : '../offical/bert_model/nezha',
    "text_transforemr_layers":12,
    "bert_cache": 'data/cache',
    "bert_seq_length": 245,
    "bert_seq_length_infer": 245,
    "bert_learning_rate": 3e-5,
    "bert_warmup_steps": 5000,
    'bert_max_steps': 30000,
    "bert_hidden_dropout_prob": 0.1,

    "frame_embedding_size": 768,
    "max_frames": 13,
    "max_frames_infer": 13,
    "vlad_cluster_size": 64,
    "vlad_groups": 8,
    "vlad_hidden_size": 1024,
    "se_ratio": 8,
    "fc_size": 512,
    
   

    
    # ============================used vision featureExtractor ========================
    'vision_model':'swin_tiny',  #option swin_tiny efficet_former van swin_base swin_v2
    
    #============================add clip loss=======================================
    #'clip':False,
    #'clip_weight':2.0,
    'fgm':True,
    #============================freeze vison backbone==============================
    'freeze':True,
    "num_top_layer":4, #cross layers
    'freeze_bert_layer':6,
    'fold':10,
        'pretrained_model_path':'/home/tione/notebook/env/code/pretrain/double_stream/5_4has_restore/4_meter_clipvit_pretrain/checkpoint/model_epoch_4_step_68750_total_loss_ 6.262.pth'
}


def parse_args():
    configs = argparse.Namespace(**config)
    return configs
