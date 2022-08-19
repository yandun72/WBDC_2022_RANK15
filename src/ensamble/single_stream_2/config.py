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
    "learning_rate": 7e-5,
    "vison_learning_rate": 1e-4,
    "classifier_learning_rate": 4e-5,
    "video_fc_lr": 5e-5,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-6,


    "bert_dir": '../../../opensource_models/macbert',
    "text_transforemr_layers":12,
    "bert_cache": 'data/cache',
    "bert_seq_length": 245,
    "bert_seq_length_infer": 245,
    "bert_learning_rate": 3e-5,
    "bert_warmup_steps": 5000,
    'bert_max_steps': 30000,
    "bert_hidden_dropout_prob": 0.1,

    "frame_embedding_size": 768,
    "max_frames": 10,
    "max_frames_infer": 10,
    "vlad_cluster_size": 64,
    "vlad_groups": 8,
    "vlad_hidden_size": 1024,
    "se_ratio": 8,
    "fc_size": 512,
    
    # ========================== Swin ===================================
    #'swin_pretrained_path':'./opensource_models/swin_tiny_patch4_window7_224.pth',
    #'swin_pretrained_path':'./inference_model/swin_base_patch4_window7_224_22kto1k.pth',
    #'swin_pretrained_path':'./inference_model/swin_base_patch4_window7_224_22k.pth',
    #'swin_pretrained_path':'./inference_model/swinv2_base_patch4_window12_192_22k.pth',

    # ========================== Van ===================================
    #'van_pretrained_path':'./opensource_models/van_base',
   
    # ========================== EfficetFormer ===================================
    #'EfficetFormer_pretrained_path':'./opensource_models/efficientformer_l1_300d.pth',

    #============================clip vit/16===========================
    "clip_vit_dir":"./inference_model/ViT-B-16.pt",
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
}


def parse_args():
    configs = argparse.Namespace(**config)
    return configs


# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")
#     parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
#     parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
#     parser.add_argument('--nprocs', default='env://', help='url used to set up distributed training')
#     parser.add_argument('--distributed', default=True, type=bool)
#     parser.add_argument("--seed", type=int, default=2022, help="random seed.")
#     parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
#     parser.add_argument('--log_name', default='macbert', help='dropout ratio')
#     # ========================= Data Configs ==========================
#     parser.add_argument('--unlabeld_annotation', type=str, default='/home/tione/notebook/data/annotations/unlabeled_new.json')
#     parser.add_argument('--unlabeld_zip_frames', type=str, default="/home/tione/notebook/data/Vit_B_16_unlabeled_zip_feats")
    
#     parser.add_argument('--train_annotation', type=str, default='/home/tione/notebook/data/annotations/labeled.json')
#     parser.add_argument('--train_zip_frames', type=str, default='/home/tione/notebook/data/Vit_B_16_unlabeled_zip_feats')
    
    

#     parser.add_argument('--test_output_csv', type=str, default='submit/result.csv')
#     parser.add_argument('--val_ratio', default=0.01, type=float, help='split 10 percentages of training data as validation')
#     parser.add_argument('--batch_size', default=48, type=int, help="use for training duration per worker")
#     parser.add_argument('--val_batch_size', default=4, type=int, help="use for validation duration per worker")
#     parser.add_argument('--test_batch_size', default=2, type=int, help="use for testing duration per worker")
#     parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
#     parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

#     # ======================== SavedModel Configs =========================
#     parser.add_argument('--savedmodel_path', type=str, default='./save')
#     parser.add_argument('--ckpt_file', type=str, default='save/model.bin')
#     parser.add_argument('--best_score', default=0.0, type=float, help='save checkpoint if mean_f1 > best_score')

#     # ========================= Learning Configs ==========================
#     parser.add_argument('--max_epochs', type=int, default=3, help='How many epochs')
#     parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
#     parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
#     parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
#     parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
#     parser.add_argument('--learning_rate', default=7e-5, type=float, help='initial learning rate')
#     parser.add_argument('--vison_learning_rate', default=8e-5, type=float, help='initial learning rate')
#     parser.add_argument('--classifier_learning_rate', default=4e-5, type=float, help='initial learning rate')
#     parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
#     parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

#     # ========================== Swin ===================================
#     parser.add_argument('--swin_pretrained_path', type=str, default='/home/tione/notebook/env/pretrainned_model/swin_tiny_patch4_window7_224.pth')
#     parser.add_argument('--output_dir', default=r'/home/tione/notebook/env/code/pretrain/double_stream/1_ALBEF/pretrain/')
#     # ========================== Title BERT =============================
#     parser.add_argument('--bert_dir', type=str, default=r'/home/tione/notebook/env/baseline/opensource_models/chinese-macbert-base')
#     parser.add_argument('--bert_cache', type=str, default='data/cache')
#     parser.add_argument('--bert_seq_length', type=int, default=245)
#     parser.add_argument('--bert_warmup_steps', type=int, default=5000)
#     parser.add_argument('--bert_max_steps', type=int, default=30000)
#     parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

#     # ========================== Video =============================
#     parser.add_argument('--frame_embedding_size', type=int, default=768)
#     parser.add_argument('--max_frames', type=int, default=16)
#     parser.add_argument('--text_transforemr_layers', type=int, default=12)
#     parser.add_argument('--vlad_cluster_size', type=int, default=64)
#     parser.add_argument('--vlad_groups', type=int, default=4)
#     parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
#     parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

#     # ========================== Fusion Layer =============================
#     parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")


#     parser.add_argument('--vision_encoder', type=str, default='swin_tiny', help="linear size before final linear")
#     return parser.parse_args()
