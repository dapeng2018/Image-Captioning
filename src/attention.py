"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Class responsible for text-guided attention.
"""

import logging
import math
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Attention:
    def __init__(self, image_encoding, caption_encoding):
        logging.info("New 'Attention' instance has been initialized.")

        with tf.name_scope('attention'):
            # Image feature ops
            image_encoding = tf.reshape(image_encoding, shape=[-1, FLAGS.kk])
            w_image_init = tf.truncated_normal([FLAGS.kk, FLAGS.conv_size], stddev=.02)
            w_image = tf.Variable(w_image_init)
            c_image = tf.matmul(image_encoding, w_image)
            b_image = tf.Variable(tf.constant(.1, shape=[FLAGS.conv_size]))
            c_image = tf.nn.bias_add(c_image, b_image)

            # Caption feature ops
            w_caption_init = tf.truncated_normal([FLAGS.stv_size, FLAGS.conv_size], stddev=.02)
            w_caption = tf.Variable(w_caption_init)
            c_caption = tf.matmul(caption_encoding, w_caption)
            c_caption = tf.concat([c_caption for _ in range(FLAGS.conv_size)], axis=0)

            # Weighted attention ops
            w_attention_init = tf.truncated_normal([FLAGS.conv_size, FLAGS.kk], stddev=.02)
            w_attention = tf.Variable(w_attention_init)
            t_attention = tf.nn.tanh(c_image + c_caption)
            e = tf.matmul(t_attention, w_attention)

            # Context ops
            self.a = tf.nn.softmax(e)
            z = self.a * image_encoding
            z = tf.reshape(z, shape=[-1, FLAGS.conv_size, FLAGS.kk])
            z = tf.reduce_sum(z, axis=2)
            self.context_vector = z
