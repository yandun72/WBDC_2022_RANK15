cd ./single_stream_1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 extract_feature_label_8frame.py #对测试集抽帧
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 inference.py#第一个单流预测
cd ..


cd ./single_stream_2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 inference.py#第二个单流预测
cd ..


cd ./double_stream_2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 inference.py #双流预训练预测
cd ..

python ensamble.py


