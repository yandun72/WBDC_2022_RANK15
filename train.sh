'''
step 1:特征抽取
'''
cd ./src/视频抽帧特征提取
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 extract_feature_label.py #对10w抽帧
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 extract_feature_unlabel.py #对100w抽帧

'''
step 2:开始预训练
'''
cd..
cd ./pretrain/single_1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrain.py #单流有预训练 qq浏览器式预训练
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrain_continue.py #在第五个epoch结束的时候中断了一次，这里恢复继续训完10个epoch，pretrain里面写了个break，模拟当时的场景

cd ..
cd ./pretrain/single_2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrain.py #单流有预训练 albef式,没有中断过，3epoch，没训充分，一个epoch要6h太久了

cd ..
cd ./pretrain/double_2
#因为这里非常吃显存，基本跑完一个epoch就oom了,当时我是分五次跑完了五个epoch
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrain.py #双流有预训练albef式
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrain_continue.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrain_continue_2.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrain_continue_3.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrain_continue_4.py



'''
开始微调
'''
cd ../../
cd ./fintunue/single_1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_quanliang.py

cd ..
cd ./fintunue/single_2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_quanliang.py

cd..
cd ./fintunue/double_2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_quanliang_ddp.py
