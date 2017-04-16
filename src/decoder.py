"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import logging
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Decoder:
    def __init__(self, context_vector):
        print("New 'Decoder' instance has been initialized.")

        with tf.variable_scope('decoder'):
            # Embedding layer
            embedding_c_init = tf.truncated_normal([FLAGS.conv_size, FLAGS.vocab_size], stddev=.1)
            embedding_c = tf.Variable(embedding_c_init)
            x0 = tf.matmul(context_vector, embedding_c)

            # LSTM layer
            lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=False)
            rnn_output, rnn_state = tf.contrib.rnn.static_rnn(lstm, [x0], dtype=tf.float32)

            # Prediction layer
            output = tf.nn.dropout(rnn_output[0], FLAGS.dropout_rate)
            self.output = tf.nn.softmax(output)

    def transcribe_caption(self):
        pass
