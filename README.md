# 代码说明
## 数据
  使用大赛提供的有标注数据（10万）
  使用无标注数据。
## 预训练模型
  使用了 huggingface 上提供的 hfl/chinese-macbert-base 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base
  使用了OpenAI开源的clip模型，链接是：https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

## 算法描述
   最终的模型是由13帧的两个不同的单流预训练和一个双流预训练简单平均融合得到
   文本特征的backbone是macbert，视觉特征backbone是clip的视觉编码器
   单流第一个预训练参考QQ浏览器式，并且为了计算itm的loss单独过了一遍transformer，预训练任务是mlm+itm+itc,训练了10epoch
   单流第二个预训练参考ALBEF的训练方式,利用了动量蒸馏去计算itc loss和mlm loss,预训练任务上多加了mfm任务，训练了3epoch,原因是时间来不及，一个epoch在6h多
   双流预训练，也是参考ALBEF，但是没有mfm任务，跑了5epoch。
   微调的时候，简单采用ema、fgm提升模型的泛化性，没有采用其他复杂思路去处理。

## 训练流程
   1 使用clip的视觉编码器对10w和100w图像数据进行抽帧和特征提取
   2 分别执行single_1 single_2 double_2的预训练任务
   3 开始微调，在对应finetune文件夹下，运行微调程序

## 测试流程
    1 在ensamble文件夹下使用single1下的特征抽取程序，对测试集图像进行抽帧和特征提取
    2 分别执行single_1 single_2 double_2的推理任务
    3 在enamble.py程序下将3个模型产生的logits矩阵进行简单的平均融合，即可生成最终的预测结果