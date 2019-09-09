# Classical2Modern

This is an interpreter which translate classical Chinese to modern Chinese.

该项目是一个把文言文翻译成现代文的翻译器。

## 问题描述

文言文是中国古代写作或交谈时使用的语言，文字和语法都与现代中文有所不同。如：
> 臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯

翻译为现代文：
> 我本来是平民，在南阳亲自耕田，在乱世中苟且保全性命，不奢求在诸侯之中出名。

大量古书文献记载，流传到今天，但并不是所有人都能轻易读懂文言文的内容，所以需要一个翻译器，能把文言文翻译成现代文，方便人们的阅读。

## 方法框架

### 训练阶段

+ 使用准备好的数据，进行简单的预处理；

+ 构建神经网络，使用Transformer神经网络，3层encoder和3层decoder，embedding维度256，feed_forward维度512，分成8头num_heads；

+ 使用char embedding，对每个字符进行编码；

+ 损失函数使用交叉熵cross_entropy损失函数（与负对数negative_log损失函数相同效果），优化器使用Adam；

### 测试阶段（单条测试、线上应用）

+ 对输入文言文进行分句处理，确保单条数据长度不会超过设定长度

+ 把切分的所有句子作为一个batch输入到模型中，输出翻译结果

## 数据集

[数据收集和预处理](data/README.md)

## 依赖

```
python3
tensorflow==1.12.0
tqdm
jieba
nltk
```

## 运行

训练

```
python train.py
```

测试

```
python test.py
```
需要下载预训练模型[请点击](https://pan.baidu.com/s/1WGJ8G8w8BU7qzTZhiuFdsw)

单条测试

```
python infer.py
```

## To Do

+ 模型调优

+ 计划将会开源数据集
