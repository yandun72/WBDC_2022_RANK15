cd ./src
mkdir Vit_B_16_unlabeled_zip_feats
mkdir ./pretrain/single_1/checkpoint
mkdir ./pretrain/single_2/checkpoint
mkdir ./pretrain/double_2/checkpoint
mkdir ./finetune/single_1/save
mkdir ./finetune/single_2/save
mkdir ./finetune/double_2/save
pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple  #训练的时候是在pytorch_py3上安装的