"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import logging
import re
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Vocab:
    def __init__(self):
        print("New 'Vocab' instance has been initialized.")

        self.list = self.get_list()
        FLAGS.vocab_size = int(self.list.get_shape()[0])

    @staticmethod
    def add_bos_eos(sentence):
        sentence.insert(0, '<bos>')
        sentence.appent('<eos>')

    @staticmethod
    def clean(sentence):
        return [re.sub(r"[^\w\s]]", "", word) for word in sentence]

    @staticmethod
    def get_list():
        filename = helpers.get_lib_path() + '/stv/vocab.txt'
        lines = [line.rstrip('\n') for line in open(filename)]
        return tf.convert_to_tensor(lines)

    def encode_one_hot(self, sentence):
        pass
