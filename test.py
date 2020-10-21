# -*- coding: utf-8 -*-

import os
import logging
import tensorflow as tf

from hparams import Hparams
from model import Transformer
from data_load import get_batch
from utils import get_hypotheses, load_hparams, calc_bleu_nltk

logging.basicConfig(level=logging.INFO)

# Loading hyper params
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)
logging.info("Hyper Params :")
logging.info("\n".join(["{} = {}".format(key.rjust(20, " "), val) for key, val in hp._get_kwargs()]))

logging.info("# Prepare test batches")
test_batches, num_test_batches, num_test_samples = get_batch(hp.test1, hp.test1,
                                                             hp.maxlen1, hp.maxlen2,
                                                             hp.vocab, hp.test_batch_size,
                                                             shuffle=False)
iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
model = Transformer(hp)

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = ckpt_ if ckpt_ else hp.ckpt
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    y_hat = model.eval(sess, test_init_op, xs, ys)

    logging.info("# get hypotheses")
    hypotheses = get_hypotheses(num_test_batches, num_test_samples, y_hat, model.idx2token)

    logging.info("# write results")
    model_output = os.path.split(ckpt)[-1]
    if not os.path.exists(hp.testdir):
        os.makedirs(hp.testdir)
    translation = os.path.join(hp.testdir, model_output)
    with open(translation, 'w', encoding="utf-8") as fout:
        fout.write("\n".join(hypotheses))

    logging.info("# calc bleu score and append it to translation")
    calc_bleu_nltk(hp.test2, translation)
