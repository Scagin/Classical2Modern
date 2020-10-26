# -*- coding: utf-8 -*-
# /usr/bin/python3

import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, \
    multihead_attention, label_smoothing, noam_scheme

logging.basicConfig(level=logging.INFO)


class Transformer:
    """
    Transformer network

    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    """

    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

        self.input_x = tf.placeholder(dtype=tf.int32, shape=(None, None), name="input_x")
        self.decoder_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="decoder_input")
        self.target = tf.placeholder(dtype=tf.int32, shape=(None, None), name="target")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")

        # encoder
        self.encoder_hidden = self.encode(self.input_x, training=self.is_training)

        # decoder
        self.logits = self.decode(self.decoder_input, self.encoder_hidden, training=self.is_training)

        self.y_hat = tf.to_int32(tf.argmax(self.logits, axis=-1), name="y_predict_v2")

        # loss
        self.smoothing_y = label_smoothing(tf.one_hot(self.target, depth=self.hp.vocab_size))
        self.ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.smoothing_y)
        nonpadding = tf.to_float(tf.not_equal(self.target, self.token2idx["<pad>"]))
        self.loss = tf.reduce_sum(self.ce_loss * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        # optimize
        self.global_step = tf.train.get_or_create_global_step()
        self.lr = noam_scheme(self.hp.lr, self.global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        # tensorboard
        tf.summary.scalar('lr', self.lr)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("global_step", self.global_step)
        self.summaries = tf.summary.merge_all()

        # predict part
        self.y_predict = tf.identity(self.greedy_search(), name="y_predict")

    def encode(self, x, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x)  # (N, T1, d_model)
            enc *= self.hp.d_model ** 0.5  # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory

    def decode(self, decoder_inputs, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings)  # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights, name="logits")  # (N, T2, vocab_size)

        return logits

    def greedy_search(self):
        decoder_inputs = tf.ones((tf.shape(self.input_x)[0], 1), tf.int32) * self.token2idx["<s>"]
        _decoder_inputs = tf.identity(decoder_inputs)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits = self.decode(_decoder_inputs, self.encoder_hidden, False)
            y_predict = tf.to_int32(tf.argmax(logits, axis=-1))
            if tf.reduce_mean(y_predict[:, -1]) == self.token2idx["<pad>"] \
                    or tf.reduce_mean(y_predict[:, -1]) == self.token2idx["</s>"]:
                break

            _decoder_inputs = tf.concat((decoder_inputs, y_predict), 1)
        return y_predict

    def beam_search(self):
        # TODO
        pass

    def eval(self, sess, eval_init_op, xs, ys, num_batches):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        logging.info("# test evaluation")
        sess.run(eval_init_op)
        result = None
        losses = []
        for _ in range(num_batches):
            _input_x, sent1, _decode_in, _tgt, sent2 = sess.run([xs[0], xs[-1], ys[0], ys[1], ys[-1]])

            _loss, y_hat = sess.run([self.loss, self.y_predict],
                                    feed_dict={self.input_x: _input_x, self.decoder_input: _decode_in,
                                               self.target: _tgt, self.is_training: False})
            losses.append(_loss)
            if result is not None:
                result = np.concatenate([result, y_hat], axis=0)
            else:
                result = y_hat

        return result, np.mean(losses)

    def infer(self, sess, input_token_ids):
        y_predict = sess.run(self.y_predict, feed_dict={self.input_x: input_token_ids, self.is_training: False})

        return y_predict

    def infer_v2(self, sess, input_token_ids):
        decoder_inputs = np.ones((1, 1), np.int32) * self.token2idx["<s>"]
        _decoder_inputs = decoder_inputs

        for _ in range(self.hp.maxlen2):
            _logits = sess.run(self.logits,
                               feed_dict={self.input_x: input_token_ids, self.decoder_input: _decoder_inputs,
                                          self.is_training: False})
            y_hat = np.argmax(_logits, -1)

            if np.mean(y_hat[:, -1]) == self.token2idx["<pad>"]:
                break

            _decoder_inputs = np.concatenate((decoder_inputs, y_hat), 1)

        return y_hat
