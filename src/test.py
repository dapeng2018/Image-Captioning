"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import tensorflow as tf
from caption_extractor import CaptionExtractor
from vgg.fcn16_vgg import FCN16VGG as Vgg16
from decoder import Decoder

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('path', None, '')

with tf.Session as sess:
    extractor = CaptionExtractor(is_training=False)
    sess.run(tf.global_variables_initializer())

    sess.close()

exit(0)
