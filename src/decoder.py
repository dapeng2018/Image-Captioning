"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import math
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Decoder:
    def __init__(self):
        print("New 'Decoder' instance has been initialized.")

    def build(self, context_vector, seq_len):
        with tf.name_scope('decoder'):
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            embedding_matrix = tf.get_variable("embedding_matrix",
                                               shape=[FLAGS.vocab_size, FLAGS.embedding_size],
                                               initializer=initializer)

            embedding = tf.nn.embedding_lookup(embedding_matrix, context_vector)
            cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(1)

            self.output, self.state = tf.nn.dynamic_rnn(cell,
                                                        dtype=tf.float32,
                                                        inputs=embedding,
                                                        sequence_length=seq_len,
                                                        swap_memory=True)
