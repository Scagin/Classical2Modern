# -*- coding: utf-8 -*-

import time
import logging
import tensorflow as tf

from model import Transformer
from hparams import Hparams
from utils import load_hparams

logging.basicConfig(level=logging.INFO)

# Loading hyper params
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)
logging.info("Hyper Params :")
logging.info("\n".join(["{} = {}".format(key.rjust(20, " "), val) for key, val in hp._get_kwargs()]))

logging.info("# Load model")
model = Transformer(hp)

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = ckpt_ if ckpt_ else hp.ckpt  # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    while True:
        text = input("请输入测试样本：")
        tokens = [ch for ch in text] + ["</s>"]
        x = [model.token2idx.get(t, model.token2idx["<unk>"]) for t in tokens]
        predict = model.infer(sess, [x])
        token_pred = [model.idx2token.get(t_id, "#") for t_id in predict[0]]
        translation = "".join(token_pred).split("</s>")[0]
        logging.info("  译文: " + translation)
        time.sleep(0.1)
