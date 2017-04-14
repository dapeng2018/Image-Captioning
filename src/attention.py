"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import logging
import tensorflow as tf


class Attention:
    def __init__(self):
        print("New 'Attention' instance has been initialized.")

    def build(self, image_encoding, caption_encoding):
        weights_image = tf.Variable(tf.random_uniform([100000, 300], -1.0, 1.0))
        weights_caption = tf.Variable(tf.random_uniform([100000, 300], -1.0, 1.0))
        weights_attention = tf.Variable(tf.random_uniform([100000, 300], -1.0, 1.0))

        weighted_image = tf.matmul(image_encoding, weights_image)
        weighted_caption = tf.matmul(caption_encoding, weights_caption)
        weighted_attention = tf.matmul(weighted_image + weighted_caption, weights_attention)

        activation = tf.nn.softmax(weighted_attention)
        self.context_vector = tf.matmul(activation, image_encoding)
