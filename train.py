# -*- coding: utf-8 -*-
# /usr/bin/python3

import os
import math
import logging
from tqdm import tqdm
import tensorflow as tf

from hparams import Hparams
from model import Transformer
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu_nltk

logging.basicConfig(level=logging.INFO)

# Loading hyper params
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)
logging.info("Hyper Params :")
logging.info("\n".join(["{} = {}".format(key.rjust(20, " "), val) for key, val in hp._get_kwargs()]))

# Data generator
logging.info("Prepare Train/Eval batches...")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2, hp.maxlen1, hp.maxlen2,
                                                                hp.vocab, hp.batch_size, shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2, 100000, 100000,
                                                             hp.vocab, hp.batch_size, shuffle=False)

# Batch iterator
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

# Build model
logging.info("Build model...")
model = Transformer(hp)
logging.info("Model is built!")

# Session
logging.info("Session initialize")
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    # Check & Load latest version model checkpoint
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(model.global_step)

    # Start training
    for i in tqdm(range(_gs, total_steps + 1)):
        _input_x, _decoder_input, _target = sess.run([xs[0], ys[0], ys[1]])
        _, _gs, _summary = sess.run([model.train_op, model.global_step, model.summaries],
                                    feed_dict={model.input_x: _input_x, model.decoder_input: _decoder_input,
                                               model.target: _target, model.is_training: True})
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        # Evaluation
        if _gs and _gs % num_train_batches == 0:
            logging.info("Epoch {} is done".format(epoch))
            _loss = sess.run(model.loss,
                             feed_dict={model.input_x: _input_x, model.decoder_input: _decoder_input,
                                        model.target: _target, model.is_training: False})

            y_hat = model.eval(sess, eval_init_op, xs, ys)

            # id to token
            logging.info("# Get hypotheses")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, y_hat, model.idx2token)

            if not os.path.exists(hp.evaldir):
                os.makedirs(hp.evaldir)
            logging.info("# Write results")
            model_output = "translation_E{:2d}L{:.2f}".format(epoch, _loss)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w', encoding="utf-8") as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# Calculate bleu score and append it to translation")

            calc_bleu_nltk(hp.eval2, translation)

            logging.info("# Save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("After training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            sess.run(train_init_op)

    summary_writer.close()

logging.info("Done")
