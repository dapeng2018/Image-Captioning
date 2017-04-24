"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Class responsible for decoding a context vector into a predicted encoded caption.
"""

import logging
import tensorflow as tf
from functools import partial

FLAGS = tf.flags.FLAGS


class Decoder:
    def __init__(self, context_vector, inputs, sequence_length=None):
        logging.info("New 'Decoder' instance has been initialized.")
        self.inputs = inputs

        with tf.name_scope('decoder'):
            # x0 (embeds attention context vector to word embedding space)
            x0_weights_init = tf.truncated_normal([FLAGS.conv_size, FLAGS.embedding_size], stddev=.1)
            x0_weights = tf.Variable(x0_weights_init)
            x0 = tf.matmul(context_vector, x0_weights)

            # Word embedding layer
            embedding_shape = [FLAGS.vocab_size, FLAGS.embedding_size]
            embedding_init = tf.random_uniform(embedding_shape, -1., 1.)
            self.word_embeddings = tf.Variable(embedding_init)

            def embed(embedding, x):
                x = tf.reshape(x, [-1])
                return tf.nn.embedding_lookup(embedding, tf.cast(x, dtype=tf.int32))

            embed_partial = partial(embed, self.word_embeddings)
            xt = tf.map_fn(embed_partial, inputs)

            # Combine context hidden with other hiddens
            x0 = tf.expand_dims(x0, axis=1)
            xt = tf.concat([x0, xt], axis=1)

            # LSTM layer
            lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            init_state = lstm.zero_state(FLAGS.batch_size, dtype=tf.float32)
            outputs, states = tf.nn.dynamic_rnn(lstm, xt, sequence_length, init_state, dtype=tf.float32)

            # Prediction layer
            def predict(embedding, x):
                return tf.matmul(x, embedding, transpose_b=True)

            predict_partial = partial(predict, self.word_embeddings)
            predictions = tf.map_fn(predict_partial, outputs)
            predictions = tf.nn.dropout(predictions, FLAGS.dropout_rate)
            predictions = tf.nn.softmax(predictions + FLAGS.epsilon)
            self.last_output = predictions[:, -1, :]
            self.outputs = predictions

    @staticmethod
    def make_readable(word_list):
        """
        Given a byte numpy array representing strings of words, decode it into a readable sentence

        :param word_list:
        :return: a string of the decoded caption
        """

        # Decode words
        word_list = [word.decode('UTF-8') for word in word_list.tolist()]

        # Get words up to <eos>
        if '<eos>' in word_list:
            eos_index = word_list.index('<eos>')
            word_list = word_list[:eos_index]

        return ' '.join(word_list)

    def sample(self, expand=True):
        """
        Randomly sample from the prediction distribution
        This is used for scheduled sampling

        :return: tensor containing indices for the sampled word embedding of shape [batch size, 1]
        """

        # Make positive and get probabilities
        positive = tf.exp(self.last_output)
        probabilities = positive / tf.reduce_sum(positive)

        # Use a randomly generated uniform distirbution in attempt to draw  random sample
        shape = [FLAGS.batch_size, tf.shape(self.last_output)[-1]]
        random = tf.random_uniform(shape, minval=0., maxval=1.)
        sample = tf.argmax(probabilities - random, axis=1)

        if expand:
            sample = tf.expand_dims(sample, axis=1)

        return sample
