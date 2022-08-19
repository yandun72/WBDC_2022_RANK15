# Pretrain file num
PRETRAIN_FILE_NUM = 15
LOAD_DATA_TYPE = 'mem'#'fluid'
# Training params
NUM_FOLDS = 1
SEED = 42
# BATCH_SIZE = 128
BATCH_SIZE = 64  #ddp模式中每个batch是16
NUM_EPOCHS = 12
WARMUP_RATIO = 0.2
REINIT_LAYER = 0
WEIGHT_DECAY = 0.01
PRETRAIN_TASK = ['mfm', 'itm','mlm']
BERT_PATH = '/home/tione/notebook/env/baseline/opensource_models/chinese-macbert-base'
NUM_TOP_LAYER = 4

