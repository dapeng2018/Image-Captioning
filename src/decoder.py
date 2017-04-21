"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Class responsible for decoding a context vector into a predicted encoded caption.
"""

import logging
import tensorflow as tf
from functools import partial

FLAGS = tf.flags.FLAGS


class Decoder:
    def __init__(self, context_vector, inputs):
        logging.info("New 'Decoder' instance has been initialized.")

        with tf.variable_scope('decoder'):
            # x0 (embeds attention context vector to word embedding space)
            x0_weights_init = tf.random_uniform([FLAGS.conv_size, FLAGS.embedding_size], -1.0, 1.0)
            x0_weights = tf.Variable(x0_weights_init)
            x0 = tf.matmul(context_vector, x0_weights)

            # Word embedding layer
            embedding_shape = [5, FLAGS.embedding_size]
            embedding_init = tf.random_uniform(embedding_shape, -1., 1.)
            self.word_embeddings = tf.Variable(embedding_init)

            def embed(embedding, x):
                return tf.matmul(x, embedding)

            _embed = partial(embed, self.word_embeddings)
            xt = tf.map_fn(_embed, inputs)

            # Combine context hidden with other hiddens
            x0 = tf.expand_dims(x0, axis=1)
            xt = tf.concat([x0, xt], axis=1)

            # LSTM layer
            lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            init_state = lstm.zero_state(FLAGS.batch_size, dtype=tf.float32)
            outputs, states = tf.nn.dynamic_rnn(lstm, xt, initial_state=init_state, dtype=tf.float32)

            # Prediction layer
            prediction = tf.matmul(outputs[-1], self.word_embeddings, transpose_b=True)
            prediction = tf.nn.dropout(prediction, FLAGS.dropout_rate)
            self.output = prediction

    def get_caption(self, vocab):
        caption = tf.convert_to_tensor(self.logits)
        caption = tf.reshape(caption, [FLAGS.max_caption_size, -1])

        max_indices = tf.argmax(caption, axis=1)
        max_indices = tf.reshape(max_indices, [-1])
        words = tf.gather(vocab.list, max_indices)

        return words

    @staticmethod
    def make_readable(word_list):
        # Decode words
        word_list = [word.decode('UTF-8') for word in word_list.tolist()]

        # Get words up to <eos>
        if '<eos>' in word_list:
            eos_index = word_list.index('<eos>')
            word_list = word_list[:eos_index]

        return ' '.join(word_list)
