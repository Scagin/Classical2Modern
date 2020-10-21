# Classical2Modern

This is an interpreter which translate classical Chinese to modern Chinese.

该项目是一个把文言文翻译成现代文的翻译器。

## 问题描述

文言文是中国古代写作或交谈时使用的语言，文字和语法都与现代中文有所不同。如：
> 臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯

翻译为现代文：
> 我本来是平民，在南阳亲自耕田，在乱世中苟且保全性命，不奢求在诸侯之中出名。

大量古书文献记载，流传到今天，但并不是所有人都能轻易读懂文言文的内容，
所以需要一个翻译器，能把文言文翻译成现代文，方便人们的阅读。

## 数据集

[数据收集和预处理](data)

> 从中挑选出部分人工整理数据，已开源，请前往[CCTC](https://github.com/Scagin/CCTC)

## 依赖

```
python3
tensorflow==1.12.0
nltk
numpy
jieba
tqdm
```

## 快速开始

- 安装依赖

```shell script
pip install -r reuirements.txt
```

- 数据准备

```
cd data

git clone https://github.com/Scagin/CCTC.git

python data_scripts.py
```

- 训练

使用默认参数训练模型

```
python train.py
```

你也可以通过修改`hparams.py`文件或者添加运行参数的形式，调整训练的超参数配置

```
python train.py --batch_size 128 \
                --lr 0.0001 \
                --num_epochs 256 \
                --d_model 512 \
                --d_ff 2048 \
                --num_blocks 6 \
                --num_heads 8

```

> 需要下载预训练模型[请点击](https://pan.baidu.com/s/1WGJ8G8w8BU7qzTZhiuFdsw)

- 测试

`--ckpt`为模型保存的文件夹路径

```
python test.py --ckpt=checkpoints/v1.0.0
```

- 单条推理

```
python infer.py --ckpt=checkpoints/v1.0.0
```

## 联系方式

 - 有任何疑问或建议，都可以通过创建 *ISSUE* 的形式提出

 - 也可以通过邮件或者QQ进行技术交流

   - 邮箱：406493851@qq.com
   
   - QQ：406493851

 - 欢迎一起交流，共同进步！


