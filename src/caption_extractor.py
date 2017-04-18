"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

from cider.cider import Cider
import helpers
import itertools
import json
import logging
import math
import nltk
import random
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

FLAGS = tf.flags.FLAGS
NUM_GRAMS = 4
NUM_NEIGHBORS = 60


class CaptionExtractor:
    def __init__(self, candidate_captions, is_training=True):
        logging.info("New 'CaptionExtractor' instance has been initialized.")

        if is_training:
            self.guidance_caption_train = candidate_captions[random.randint(0, FLAGS.k)]
        else:
            self.cider = Cider(n=FLAGS.ngrams)

            candidate_captions_eval = candidate_captions.eval()
            hyps, refs = {}, {}

            for i, caption in candidate_captions_eval:
                hyps[i] = caption
                refs[i] = []
                for reference in candidate_captions:
                    refs[i].append(reference)

            score, scores = Cider.compute_score(hyps, refs)

            index = 0

            self.guidance_caption_test = candidate_captions_eval[index]
