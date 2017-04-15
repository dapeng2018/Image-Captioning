"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import logging
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Attention:
    def __init__(self):
        print("New 'Attention' instance has been initialized.")

    def build(self, image_encoding, caption_encoding):
        w_image_init = tf.random_uniform([FLAGS.batch_size, 32, 32, 512], -1., 1.)
        w_image = tf.Variable(w_image_init)

        w_caption_init = tf.random_uniform(tf.shape(caption_encoding), -1., 1.)
        w_caption = tf.Variable(w_caption_init)

        w_attention = tf.Variable(tf.random_uniform([100000, 300]))

        c_image = tf.matmul(image_encoding, w_image, transpose_b=True)
        c_caption = tf.matmul(caption_encoding, w_caption, transpose_b=True)
        c_attention = tf.matmul(c_image + c_caption, w_attention)

        activation = tf.nn.softmax(c_attention)
        self.context_vector = tf.matmul(activation, image_encoding)
