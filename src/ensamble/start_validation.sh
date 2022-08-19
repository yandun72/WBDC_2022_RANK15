# cd ./single_stream_1
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 validation_1.py
# cd ..

# cd ./single_stream_2
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 validation_2.py
# cd ..

# cd ./double_stream
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 validation_3.py
# cd ..

# cd ./double_stream_2
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 validation_4.py
# cd ..

python ensamble_validation.py
date -R