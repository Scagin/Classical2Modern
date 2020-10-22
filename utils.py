# -*- coding: utf-8 -*-
# /usr/bin/python3

import os
import json
import jieba
import logging
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

logging.basicConfig(level=logging.INFO)


def calc_num_batches(total_num, batch_size):
    """
    Calculates total number of batches.
    """
    return total_num // batch_size + int(total_num % batch_size != 0)


def postprocess(hypotheses, idx2token):
    """
    Processes translation outputs.
    """
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ")  # remove bpe symbols
        _hypotheses.append(sent.strip())
    return _hypotheses


def save_hparams(hparams, path):
    """
    Saves hparams to path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)


def load_hparams(parser, path):
    """
    Loads hparams and overrides parser
    """
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v


def save_variable_specs(fpath):
    """
    Saves information about variables such as
    their name, shape, and total parameter number
    """
    def _get_size(shp):
        size = 1
        for d in range(len(shp)):
            size *= shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    logging.info("num_params: {}".format(num_params))
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")


def get_hypotheses(num_batches, num_samples, predict, dict):
    hypotheses = []
    for _ in range(num_batches):
        hypotheses.extend(predict.tolist())
    hypotheses = postprocess(hypotheses, dict)
    return hypotheses[:num_samples]


def calc_bleu_nltk(ref, translation):
    '''Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path

    Returns
    translation that the bleu score is appended to'''
    ref_lines = [[jieba.lcut(line.strip())] for line in open(ref, encoding="utf-8") if line.strip()]
    trans_lines = [jieba.lcut(line.strip()) for line in open(translation, encoding="utf-8") if line.strip()]
    bleu_score_report = corpus_bleu(ref_lines, trans_lines)
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
