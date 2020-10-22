# -*- coding: utf-8 -*-
# /usr/bin/python3

import os
import time
import math
import logging
from tqdm import tqdm
import tensorflow as tf

from hparams import Hparams
from model import Transformer
from data_load import get_batch, load_vocab
from utils import load_hparams, save_hparams, save_variable_specs, get_hypotheses, calc_bleu_nltk

logging.basicConfig(level=logging.INFO)
__version__ = "v1.1.0"


def infer(hp):
    load_hparams(hp, hp.ckpt)

    # latest checkpoint
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = ckpt_ if ckpt_ else hp.ckpt

    # load graph
    saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    graph = tf.get_default_graph()

    # load tensor
    input_x = graph.get_tensor_by_name("input_x:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    y_predict = graph.get_tensor_by_name("y_predict:0")

    # vocabulary
    token2idx, idx2token = load_vocab(hp.vocab)

    logging.info("# Session")
    with tf.Session() as sess:
        saver.restore(sess, ckpt)

        while True:
            text = input("请输入测试样本：")

            # tokens to ids
            tokens = [ch for ch in text] + ["</s>"]
            x = [token2idx.get(t, token2idx["<unk>"]) for t in tokens]

            # run calculation
            predict_result = sess.run(y_predict, feed_dict={input_x: [x], is_training: False})

            # ids to tokens
            token_pred = [idx2token.get(t_id, "#") for t_id in predict_result[0]]
            translation = "".join(token_pred).split("</s>")[0]

            logging.info("  译文: {}".format(translation))

            time.sleep(0.1)


def test(hp):
    # Loading hyper params
    load_hparams(hp, hp.ckpt)

    logging.info("# Prepare test batches")
    test_batches, num_test_batches, num_test_samples = get_batch(hp.test1, hp.test1,
                                                                 100000, 100000,
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


def train(hp):
    save_hparams(hp, hp.checkpoints_dir)
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
        ckpt = tf.train.latest_checkpoint(hp.checkpoints_dir)
        if ckpt is None:
            logging.info("Initializing from scratch")
            sess.run(tf.global_variables_initializer())
            save_variable_specs(os.path.join(hp.checkpoints_dir, "specs"))
        else:
            saver.restore(sess, ckpt)

        summary_writer = tf.summary.FileWriter(hp.checkpoints_dir, sess.graph)

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
                ckpt_name = os.path.join(hp.checkpoints_dir, model_output)
                saver.save(sess, ckpt_name, global_step=_gs)
                logging.info("After training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

                sess.run(train_init_op)

        summary_writer.close()

    logging.info("Done")


def export_model(hp):
    """
    export model checkpoint to pb file
    """
    load_hparams(hp, hp.ckpt)

    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = ckpt_ if ckpt_ else hp.ckpt

    saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    graph = tf.get_default_graph()

    input_x = graph.get_tensor_by_name("input_x:0")
    decoder_input = graph.get_tensor_by_name("decoder_input:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    y_predict = graph.get_tensor_by_name("y_predict:0")

    with tf.Session() as sess:
        saver.restore(sess, ckpt)  # restore graph

        builder = tf.saved_model.builder.SavedModelBuilder(hp.export_model_dir)
        inputs = {'input': tf.saved_model.utils.build_tensor_info(input_x),
                  'decoder_input': tf.saved_model.utils.build_tensor_info(decoder_input),
                  'is_training': tf.saved_model.utils.build_tensor_info(is_training)}
        outputs = {'y_predict': tf.saved_model.utils.build_tensor_info(y_predict)}

        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'signature')
        builder.add_meta_graph_and_variables(sess, ['classical2modern'], {'signature': signature})
        builder.save()


def _check_version(hp=None):
    logging.info("Hello! This is Classical2Modern. You are now using the application version {}".format(__version__))


if __name__ == '__main__':
    mode_func = {"train": train, "test": test, "infer": infer, "version": _check_version, "export": export_model}

    hparams = Hparams()
    hp = hparams.get_params()
    mode = hp.mode

    func = mode_func.get(mode)

    if func:
        func(hp)
    else:
        logging.error("Sorry, you set a wrong mode not in [train, test, infer, version, export]. "
                      "Check you arguments please.")
