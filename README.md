# Classical2Modern

![Travis building status](https://img.shields.io/travis/scagin/Classical2Modern/master)
[![License](https://img.shields.io/github/license/scagin/Classical2Modern)](https://github.com/Scagin/Classical2Modern/blob/master/LICENSE)
![Stars](https://img.shields.io/github/stars/scagin/Classical2Modern)
![Forks](https://img.shields.io/github/forks/scagin/Classical2Modern)

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
tensorflow>=1.11.0
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

- 环境测试

你可以通过简单的运行`main.py`来测试你的依赖配置是否已经安装完成

```shell script
python src/main.py --mode version
```

- 本地调试

`--ckpt`为模型保存的文件夹路径

> 需要下载预训练模型[请点击链接](https://pan.baidu.com/s/1fjVMbSDtTqgWYVhBdiRBWQ) (提取码: nv7v)

```shell script
python src/main.py --mode test --ckpt=checkpoints/v1.0.0
```

#### 重新训练

- 数据准备

```
cd data

git clone https://github.com/Scagin/CCTC.git

python data_scripts.py
```

- 训练

使用默认参数训练模型

```
python src/main.py
```

你也可以通过修改`hparams.py`文件或者添加运行参数的形式，调整训练的超参数配置

```
python src/main.py  --mode train \
                    --batch_size 128 \
                    --lr 0.0001 \
                    --num_epochs 256 \
                    --d_model 512 \
                    --d_ff 2048 \
                    --num_blocks 6 \
                    --num_heads 8
```

- 测试

```
python src/main.py --mode test --ckpt=checkpoints/v1.0.0
```

## 模型API服务部署

### 指南

你可以通过 **下载Release版本** 或者 **从源码编译** 的方式使用 `REST API` 服务

下载编译后的 `zip` 文件，可以在指定平台中直接运行

v1.1版本：[下载](https://github.com/Scagin/Classical2Modern/releases/download/v1.1.0/release_v1.1.0.zip)

#### 发布版本

已发布`v1.1.0`版本，仅支持`amd64` 操作系统构架。

支持以下操作系统：

```
1. Windows
2. Linux
3. Mac Os
4. FreeBSD
```

- 启动服务

Linux
```
chmod +x ./bin/start.sh ./bin/classical2modern
./bin/start.sh -port 9391 -max_length 120 -vocab_path ./data/vocab_char.txt -model_path ./mymodel
```

`port` API 服务监听端口

`max_length` 译文允许的最大长度

`vocab_path` 字典文件路径（请使用训练模型时的字典文件）

`model_path` 模型文件夹路径

- 查看api日志

```shell script
tail -f logs/api_server.log
```

#### 源码编译

你需要
1. 配置 `Go` 运行环境

2. 安装好`tensorflow`的 `C` 和 `Go` API 依赖

3. 下载[预训练模型](https://pan.baidu.com/s/1fjVMbSDtTqgWYVhBdiRBWQ) (提取码: nv7v) 

4. 并执行`python src/main.py --mode export --ckpt v1.0.0`命令导出模型

5. 执行 `cd api_server & go build -o ../bin/classical2modern_api`

6. 启动服务
```
./classical2modern  -port 9391 -max_length 120 -vocab_path ./data/vocab_char.txt -model_path ./mymodel
```

## 协议

本项目采用[MIT开源协议](./LICENSE)，一切声明均在协议中。

## 联系方式

 - 有任何疑问或建议，都可以通过创建 *ISSUE* 的形式提出

 - 也可以通过邮件或者QQ进行技术交流

   - 邮箱：406493851@qq.com
   
   - QQ：406493851

 - 欢迎一起交流，共同进步！


