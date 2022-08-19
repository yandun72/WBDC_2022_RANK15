# Pretrain file num
PRETRAIN_FILE_NUM = 15
LOAD_DATA_TYPE = 'mem'#'fluid'
# Training params
NUM_FOLDS = 1
SEED = 42
# BATCH_SIZE = 128
BATCH_SIZE = 44
NUM_EPOCHS = 5
WARMUP_RATIO = 0.15
REINIT_LAYER = 0
WEIGHT_DECAY = 0.01
LR = {'others':6e-5, 'roberta':6e-5, 'newfc_videoreg':6e-5}
LR_LAYER_DECAY = 1.0
# PRETRAIN_TASK = ['tag', 'mlm', 'mfm']
# mlm
PRETRAIN_TASK = ['mlm','itm','mfm','itc']
