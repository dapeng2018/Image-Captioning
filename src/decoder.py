"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Decoder:
    def __init__(self, context_vector):
        print("New 'Decoder' instance has been initialized.")

        cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(512)
        output, self.state = tf.contrib.rnn.static_rnn(cell, [context_vector], dtype=tf.float32)
        self.output = tf.nn.dropout(output, FLAGS.dropout_rate)

