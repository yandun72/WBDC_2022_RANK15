# 代码说明
2022微信大数据技术挑战赛复赛top16

链接:https://algo.weixin.qq.com
## 数据
  使用大赛提供的有标注数据（10万）
  使用无标注数据。
## 预训练模型
  使用了 huggingface 上提供的 hfl/chinese-macbert-base 模型
  链接： https://huggingface.co/hfl/chinese-macbert-base
  
  使用了OpenAI开源的clip模型
  链接：https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

## 算法描述
   最终的模型是由13帧的两个不同的单流预训练和一个双流预训练简单平均融合得到
   文本特征的backbone是macbert，视觉特征backbone是clip的视觉编码器
   单流第一个预训练参考QQ浏览器式，并且为了计算itm的loss单独过了一遍transformer，预训练任务是mlm+itm+itc,训练了10epoch
   单流第二个预训练参考ALBEF的训练方式,利用了动量蒸馏去计算itc loss和mlm loss,预训练任务上多加了mfm任务，训练了3epoch,原因是时间来不及，一个epoch在6h多
   双流预训练，也是参考ALBEF，但是没有mfm任务，跑了5epoch。
   微调的时候，简单采用ema、fgm提升模型的泛化性，没有采用其他复杂思路去处理。
## 训练流程
    1 使用clip的视觉编码器对10w和100w图像数据进行抽帧和特征提取
    2 分别执行src/pretrain下的single_1 single_2 double_2的预训练任务
    3 开始微调，在对应finetune文件夹下，运行微调程序
    运行train.sh,一键执行训练集的抽帧、预训练及微调
## 推理流程
    1 在ensamble文件夹下使用single1下的特征抽取程序，对测试集图像进行抽帧和特征提取
    2 分别执行single_1 single_2 double_2的推理任务
    3 在enamble.py程序下将3个模型产生的logits矩阵进行简单的平均融合，即可生成最终的预测结果
    运行inference.sh,一键执行测试集的抽帧、推理、融合
### 加速方面
1:在抽特征的时候,如果采用ddp或者dp进行的话，会产生通信开销，因此我采取的策略是利用ddp的多进程，使rank0进程的GPU只负责提取前一半数据,使rank1进程的GPU只负责提取后一半数据，这样就可以完全避免掉抽特征的开销，不同点是会创建两个zip文件，第一个zip文件放置前一半视频帧的特征。
2：采用pytorch官网提供的混合精度训练
### 学习率
在预训练和微调都使用了分层学习率，单流的话就仅仅对bert不同层进行分层；双流的话多对双流的transformers也分层。不过写的时候发现写的有点冗余。以后会参考强哥写的:https://mp.weixin.qq.com/s/CDIeTz6ETpdQEzjdeBW93g
### 关于端到端还是分阶段
我采取的是分阶段，之前试过端到端效果不好，后面赛后听大佬说是因为视觉编码器的学习率要设到文本的1/10才行，所以估计前排还是有大佬用的端到端。
### 关于100w的数据利用
郭大赛后分享用的是伪标签，这块应该是郭大的核心武器了，我们菜鸡完全以为会没用。
### 关于陪跑大佬的trick
陪跑大佬线下观察了top5的准确率，基本85%以上？记不得了，所以他为了将小分类的准确度提升，强行将小分类的logits进行放大。大概提分4k多。等陪跑大佬的开源
