import argparse

config = {
    'log_name':'macbert',
    "seed" : 2012,
    "dropout": 0.25,
    "train_annotation" : '../baseline/demo_data/annotations/semi_demo.json',
    "train_annotation2":'../baseline/demo_data/annotations/semi_demo.json',
    "test_annotation" : "/chenqs_ms/VX_race_data/annotations/test_a.json",
    "train_zip_feats" : '../baseline/demo_data/zip_frames/demo/',
    "train_zip_feats2" : '../baseline/demo_data/zip_frames/demo/',
    "test_zip_feats" : '/chenqs_ms/VX_race_data/zip_feats/test_a.zip',
    "test_output_csv" : '/chenqs_ms/VX_race_data/base_result.csv',
    
    "val_ratio" : 0.00001,#验证集大小
    "batch_size" : 1,
    "val_batch_size" : 32,
    "test_batch_size" : 32*8,
    "prefetch" : 16,
    "num_workers" : 4,
    "savedmodel_path" : '/chenqs_ms/xy/pretrain_job/fintune_5fold/9_5的换cls为mean_pool/save/v1',
    "ckpt_file" : './save/v1/model_.bin',
    
    "best_score" : 0.5,
    'max_epochs':5,
    "max_steps" : 14000,
    "print_steps" : 100,
    "warmup_steps" : 1000,
    "minimum_lr" : 0.,
    "learning_rate" : 5e-5,
    "weight_decay" : 0.01,
    "adam_epsilon" : 1e-6,
    
    "bert_dir" : '/home/tione/notebook/env/baseline/opensource_models/chinese-macbert-base',
    #"bert_dir" : '../offical/bert_model/nezha',
    "bert_cache" : 'data/cache',
    "bert_seq_length" : 145,
    "bert_learning_rate" : 3e-5,
    "bert_warmup_steps" : 5000,
    'bert_max_steps': 30000,
    "bert_hidden_dropout_prob" : 0.1,
    
    "frame_embedding_size" : 768,
    "max_frames" : 12,
    "vlad_cluster_size" : 64,
    "vlad_groups" : 8,
    "vlad_hidden_size" : 1024,
    "se_ratio" : 8,
    "fc_size" : 512,

    # ========================== Swin ===================================
    'swin_pretrained_path':'/home/tione/notebook/env/pretrainned_model/swin_tiny_patch4_window7_224.pth',

    # ========================== Levit ===================================
    'levit_model':'256',
    'levit_pretrained_path_256':'../baseline/opensource_models/LeViT-256-13b5763e.pth',
    'levit_pretrained_path_384':'../baseline/opensource_models/LeViT-384-9bdaf2e2.pth',
    'levit_pretrained_path_192':'../baseline/opensource_models/LeViT-192-92712e41.pth',

    # ========================== EfficetFormer ===================================
    'EfficetFormer_pretrained_path':'../baseline/opensource_models/efficientformer_l3_300d.pth',

    # ============================used vision featureExtractor ========================
    'vision_model':'swin_tiny'  #option swin_tiny efficet_former levit
}
def parse_args():
    configs = argparse.Namespace(**config)
    return configs
