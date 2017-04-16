"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import tensorflow as tf
import helpers
from caption_extractor import CaptionExtractor
from decoder import Decoder
from vgg.fcn16_vgg import FCN16VGG as Vgg16


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('path', None, '')
helpers.config_model_flags()
helpers.config_logging(env='testing')


with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    sess.close()

exit(0)
