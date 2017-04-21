"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Class responsible for text-guided attention.
"""

import logging
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Attention:
    def __init__(self, image_encoding, caption_encoding):
        logging.info("New 'Attention' instance has been initialized.")

        with tf.name_scope('attention'):
            # Image feature ops
            image_encoding = tf.reshape(image_encoding, shape=[-1, FLAGS.conv_size, FLAGS.kk])
            image_encoding = tf.reduce_mean(image_encoding, axis=2)
            w_image_init = tf.random_uniform([FLAGS.conv_size, FLAGS.embedding_size], -1., 1.)
            w_image = tf.Variable(w_image_init)
            c_image = tf.matmul(image_encoding, w_image)
            b_image = tf.Variable(tf.constant(.1, shape=[FLAGS.embedding_size]))
            c_image = tf.nn.bias_add(c_image, b_image)

            # Caption feature ops
            w_caption_init = tf.random_uniform([FLAGS.stv_size, FLAGS.conv_size], -1., 1.)
            w_caption = tf.Variable(w_caption_init)
            c_caption = tf.matmul(caption_encoding, w_caption)

            # Weighted attention ops
            w_attention = tf.Variable(tf.random_uniform([FLAGS.conv_size, FLAGS.conv_size], -1., 1.))
            t_attention = tf.nn.tanh(c_image + c_caption)
            c_attention = tf.matmul(t_attention, w_attention)

            # Context ops
            alphas = tf.nn.softmax(c_attention)
            context = alphas * image_encoding
            self.context_vector = context
