"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import numpy as np
import operator
import os
import tensorflow as tf
from vgg.fcn16_vgg import FCN16VGG as Vgg16

HEIGHT = 224
WIDTH = 224
TRAINING_SIZE = helpers.get_training_size()


class Neighbor:
    def __init__(self):
        print("New 'Neighbor' instance has be initialized.")

    @staticmethod
    def nearest(image, k=60):
        """

        :param image:
        :param k:
        :return:
        """

        neighbors = {}

        image = tf.image.resize_area(image, [HEIGHT, WIDTH])
        image_shape = [1, HEIGHT, WIDTH, 3]
        image_vgg = Vgg16()
        image_vgg.build(image, image_shape[1:])
        image_encoding = image_vgg.fc7

        neighbor_img_placeholder = tf.placeholder(dtype=tf.float32, shape=image_shape)
        neighbor_vgg = Vgg16()
        neighbor_vgg.build(neighbor_img_placeholder, image_shape[1:])
        neighbor_encoding = neighbor_vgg.fc8

        for i in range(helpers.get_training_size()):
            neighbor_path = helpers.get_training_next_path()
            neighbor_filename = os.path.basename(neighbor_path)
            neighbor_image = helpers.load_image2(neighbor_path, HEIGHT, WIDTH)
            neighbor_image = neighbor_image.reshape(image_shape).astype(np.float32)
            _neighbor_encoding = neighbor_encoding.eval(feed_dict={neighbor_img_placeholder: neighbor_image})
            similarity = helpers.get_cosine_similarity(image_encoding, tf.convert_to_tensor(_neighbor_encoding))

            if len(neighbors.keys()) < k:
                neighbors[neighbor_filename] = similarity
            elif Neighbor.should_update_neighbors(neighbors, similarity):
                neighbors[neighbor_filename] = similarity
                neighbors = sorted(neighbors.items(), key=operator.itemgetter(1))[:60]

        return neighbors

    @staticmethod
    def should_update_neighbors(neighbors, similarity):
        return len([s for s in neighbors.keys() if s < similarity]) > 0