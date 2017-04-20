"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Class responsible for decoding a context vector into a predicted encoded caption.
"""

import logging
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Decoder:
    def __init__(self, context_vector):
        logging.info("New 'Decoder' instance has been initialized.")

        with tf.variable_scope('decoder'):
            # x0 (embeds attention context vector to word embedding space)
            x0_weights_init = tf.random_uniform([FLAGS.conv_size, FLAGS.embedding_size], -1.0, 1.0)
            x0_weights = tf.Variable(x0_weights_init)
            x0 = tf.matmul(context_vector, x0_weights)

            # Word embedding layer
            weights_shape = [5, FLAGS.embedding_size]
            weights_init = tf.random_uniform(weights_shape, -1., 1.)
            self.word_embeddings = tf.Variable(weights_init)
            self.logits = [tf.matmul(x0, self.word_embeddings, transpose_b=True)]

            # LSTM layer
            lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            state = lstm.zero_state(FLAGS.batch_size, dtype=tf.float32)

            for i in range(FLAGS.state_size - 1):
                w = self.logits[-1]
                x = tf.matmul(w, self.word_embeddings)
                output, state = lstm(x, state)

                # Prediction layer
                prediction = tf.matmul(output, self.word_embeddings, transpose_b=True)
                prediction = tf.nn.dropout(prediction, FLAGS.dropout_rate)
                self.logits.append(prediction)

                # Reuse the variables generated within the LSTM
                if i == 0:
                    tf.get_variable_scope().reuse_variables()

    def get_caption(self, vocab):
        caption = tf.convert_to_tensor(self.logits)
        max_indices = tf.argmax(caption, axis=1)
        words = tf.gather(vocab.list, max_indices)
        return words
