"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import numpy as np
import operator
import os
import pickle
import tensorflow as tf
from vgg.fcn16_vgg import FCN16VGG as Vgg16


class Neighbor:
    def __init__(self):
        print("New 'Neighbor' instance has be initialized.")

    @staticmethod
    def nearest(image_filename, k=60):
        neighbors = []
        return neighbors

    @staticmethod
    def append_fc7_encoding(x):
        filename = 'fc7_encoding.pkl'
        path = helpers.get_lib_path() + filename
        with open(path, 'ab') as out:
            pickle.dump(x, out)

    def make_training_fc7_file(self):
        shape = [1, 224, 224, 3]
        with tf.Session() as sess:
            image_placeholder = tf.placeholder(dtype=tf.float32, shape=shape)

            vgg = Vgg16()
            vgg.build(image_placeholder, shape[1:])
            layer = vgg.fc7

            sess.run(tf.global_variables_initializer())

            for _ in range(helpers.get_training_size()):
                filename = helpers.get_training_next_path()
                img = helpers.load_image_to(filename, shape[1], shape[2])
                img = img.reshape(shape).astype(np.float32)
                encoding = sess.run(layer, feed_dict={image_placeholder: img})
                self.append_fc7_encoding({filename: encoding})
