"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import logging
import re
import tensorflow as tf
from functools import partial

FLAGS = tf.flags.FLAGS


class Vocab:
    def __init__(self):
        logging.info("New 'Vocab' instance has been initialized.")

        self.list, self._list = self.get_list()
        FLAGS.vocab_size = 1#int(self.list.get_shape()[0])

    @staticmethod
    def add_bos_eos(sentence):
        sentence.insert(0, '<bos>')
        sentence.appent('<eos>')
        return sentence

    @staticmethod
    def clean(sentence):
        return [re.sub(r"[^\w\s]]", "", word) for word in sentence]

    @staticmethod
    def extend_to_state_size(x):
        x.extend(['' for _ in range(FLAGS.state_size - len(x))])

    @staticmethod
    def get_list():
        filename = helpers.get_lib_path() + '/stv/vocab.txt'
        lines = [line.rstrip('\n') for line in open(filename)]
        return tf.convert_to_tensor(lines), lines

    def get_bos_rnn_input(self):
        one_hot = self.get_bos_1hot()
        return [[one_hot]]

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
