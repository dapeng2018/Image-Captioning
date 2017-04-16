"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: This class is simply responsible for determining the nearest neighboring images of an image
"""

import helpers
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Neighbor:
    def __init__(self, input_encoding, training_encodings, training_filenames):
        print("New 'Neighbor' instance has be initialized.")

        similarities = helpers.get_cosine_similarity(input_encoding, training_encodings)
        _, indices = tf.nn.top_k(similarities, k=FLAGS.k)
        self.nearest = tf.gather(training_filenames, indices)
