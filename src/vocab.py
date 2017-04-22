"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import logging
import numpy as np
import re
import tensorflow as tf
from functools import partial

FLAGS = tf.flags.FLAGS


class Vocab:
    def __init__(self):
        logging.info("New 'Vocab' instance has been initialized.")

        self.list, self._list = self.get_list()
        FLAGS.vocab_size = int(self.list.get_shape()[0])

    @staticmethod
    # Insert and append the <bos> and <eos> tokens into a sentence (list of words)
    def add_bos_eos(sentence):
        sentence.insert(0, '<bos>')
        sentence.append('<eos>')
        return sentence

    @staticmethod
    # Remove anything that is not a character or space in a sentence (list of words)
    def clean(sentence):
        return [re.sub(r"[^\w\s]]", "", word) for word in sentence]

    @staticmethod
    # Append the <pad> token up to the globally specified max caption size
    def pad(x):
        x.extend(['<pad>' for _ in range(FLAGS.max_caption_size - len(x))])

    @staticmethod
    def get_list():
        filename = helpers.get_lib_path() + '/stv/vocab.txt'
        lines = [line.rstrip('\n') for line in open(filename)]
        return tf.convert_to_tensor(lines), lines

    # Return a numpy array of one-hot encoded vectors representing <bos> (to be used as the first RNN input)
    def get_bos_rnn_input(self, batch_size):
        one_hot = self.get_bos_1hot()
        batch_one_hot = [[one_hot] for _ in range(batch_size)]
        return np.array(batch_one_hot)

    # Get the one-hot encoded representation of the <bos> token
    def get_bos_1hot(self):
        index = self.get_index_from_word('<bos>')
        return helpers.index_to_1hot(index)

    # Return the index form the vocab list given a word (string)
    def get_index_from_word(self, word):
        if word in self._list:
            return self._list.index(word)
        else:
            return -1

    # Given a list of words, return their indices in the corpus
    def get_word_ids(self, words):
        def get_id(x, y):
            one_hot = tf.cast(tf.equal(x, y), tf.float32)
            return tf.argmax(one_hot)

        equals = partial(get_id, self.list)
        return tf.map_fn(equals, words)

    # Given a batch of tokenized word labels, convert them to their one hot representations
    def word_labels_to_1hot(self, batch_labels):
        batch_one_hot = []

        for labels in batch_labels:
            self.pad(labels)
            indices = [self.get_index_from_word(word) for word in labels]
            one_hot = [helpers.index_to_1hot(index) for index in indices]
            batch_one_hot.append(one_hot)

        return batch_one_hot
