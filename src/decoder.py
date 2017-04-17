"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import logging
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Decoder:
    def __init__(self, context_vector):
        logging.info("New 'Decoder' instance has been initialized.")

        with tf.variable_scope('decoder'):
            # Embedding layer
            x0_init = tf.truncated_normal([FLAGS.conv_size, FLAGS.vocab_size], stddev=.1)
            x0_c = tf.Variable(x0_init)
            x0 = tf.matmul(context_vector, x0_c)

            embed_init = tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_size], -1.0, 1.0)
            embed = tf.Variable(embed_init)
            word_ids = tf.argmax(x0, axis=1)
            word_embeddings = tf.nn.embedding_lookup(embed, word_ids)

            # LSTM layer
            lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.embedding_size, state_is_tuple=True)
            rnn_outputs, rnn_states = tf.contrib.rnn.static_rnn(
                cell=lstm,
                inputs=[word_embeddings],
                initial_state=lstm.zero_state(FLAGS.batch_size, dtype=tf.float32),
                dtype=tf.float32,
                sequence_length=tf.convert_to_tensor([FLAGS.embedding_size for _ in range(FLAGS.batch_size)]))

            # Prediction layer
            weights_shape = [FLAGS.embedding_size, FLAGS.vocab_size]
            weights_init = tf.random_uniform(weights_shape, -1., 1.)
            weights = tf.Variable(weights_init)

            def predict(output):
                return tf.matmul(output[0], weights)

            predictions = tf.map_fn(predict, rnn_outputs)
            self.logits = tf.nn.dropout(predictions, FLAGS.dropout_rate)

    def get_caption(self, vocab, caption):
        pass

    @staticmethod
    def transcribe_caption(vocab, word_indices):
        words = tf.gather(vocab.list, word_indices)
        word_list = words.eval()
        return ' '.join(word_list)
