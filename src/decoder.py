"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import logging
import tensorflow as tf
from functools import partial

FLAGS = tf.flags.FLAGS


class Decoder:
    def __init__(self, context_vector, labels):
        logging.info("New 'Decoder' instance has been initialized.")

        with tf.variable_scope('decoder'):
            # Context embedding layer
            x0_weights_init = tf.random_uniform([FLAGS.conv_size, FLAGS.vocab_size], -1.0, 1.0)
            x0_weights = tf.Variable(x0_weights_init)
            x0 = tf.matmul(context_vector, x0_weights)

            # Word embedding layer
            weights_shape = [FLAGS.vocab_size, FLAGS.embedding_size]
            weights_init = tf.random_uniform(weights_shape, -1., 1.)
            self.word_embeddings = tf.Variable(weights_init)
            x = tf.matmul(x0, self.word_embeddings)

            # LSTM layer
            lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            state = lstm.zero_state(FLAGS.batch_size, dtype=tf.float32)
            self.probabilities = []
            self.total_loss = 0.

            for i in range(FLAGS.state_size):
                x, state = lstm(x, state)

                # Prediction layer
                prediction = tf.matmul(x, self.word_embeddings, transpose_b=True)
                prediction = tf.nn.dropout(prediction, FLAGS.dropout_rate)
                self.probabilities.append(tf.nn.softmax(prediction))
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.probabilities)
                self.total_loss += tf.reduce_mean(loss)

                # Reuse the variables generated within the LSTM
                if i == 0:
                    tf.get_variable_scope().reuse_variables()

    def get_caption(self, vocab, caption):
        pass

    def get_word_embedding(self, word_ids):
        return tf.nn.embedding_lookup(self.word_embeddings, word_ids)

    def reset(self):
        self.probabilities = []
        self.total_loss = 0.
