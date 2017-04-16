"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Executable script for generating a caption given an input image and an already trained model.
"""

import logging
import helpers
import os
import tensorflow as tf
from caption_extractor import CaptionExtractor
from decoder import Decoder
from vgg.fcn16_vgg import FCN16VGG as Vgg16


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('input', None, '')
tf.flags.DEFINE_string('model_path', None, '')
helpers.config_model_flags()
helpers.config_logging(env='testing')


with tf.Session as sess:
    sess.run(tf.global_variables_initializer())

    #

    # Restore previously trained model
    saved_path = os.path.abspath(FLAGS.model_path)
    saver = tf.train.Saver()
    saver.restore(sess, saved_path)

    sess.close()

exit(0)
