"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import tensorflow as tf
from caption import Caption
from caption_encoder import CaptionEncoder
from custom_vgg16 import Vgg16

TRAINING_ITERS = 100
TRAIN_HEIGHT = 512
TRAIN_WIDTH = 512

with tf.Session() as sess:
    image_shape = [1, TRAIN_HEIGHT, TRAIN_WIDTH, 3]

    # Initialize system instances
    captioner = Caption()
    stv = CaptionEncoder()
    vgg = Vgg16()

    # Initialize placeholders
    caption_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None])
    seq_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, ])
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=image_shape)

    # Build network architectures
    stv.build(caption_placeholder, seq_len_placeholder)
    vgg.build(image_placeholder, image_shape[1:])

    #
    generated_caption = stv.output
    image_encoding = vgg.conv5_3

    #
    sess.run(tf.global_variables_initializer())

    # Optimize
    for i in range(TRAINING_ITERS):
        pass
