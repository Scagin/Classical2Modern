# -*- coding: utf-8 -*-

import os

import tensorflow as tf

from data_load import get_batch, encode
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, load_hparams, calc_bleu_nltk
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)

input_tokens = tf.placeholder(tf.int32, shape=(1, None))
xs = (input_tokens, None, None)

logging.info("# Load model")
m = Transformer(hp)
y_hat = m.infer(xs)

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    while True:
        text = input("请输入测试样本：")
        # x = encode(text, "x", m.token2idx)
        tokens = [ch for ch in text] + ["</s>"]
        x = [m.token2idx.get(t, m.token2idx["<unk>"]) for t in tokens]
        pred = sess.run(y_hat, feed_dict={input_tokens: [x]})
        token_pred = [m.idx2token.get(t_id, "#") for t_id in pred[0]]
        translation = "".join(token_pred).split("</s>")[0]
        logging.info("  译文: " + translation)

