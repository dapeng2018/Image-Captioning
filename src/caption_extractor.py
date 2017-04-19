"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

from cider.cider import Cider
import helpers
import itertools
import json
import logging
import nltk
import numpy as np
import os
import random
import re
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from threading import Thread, Lock

FLAGS = tf.flags.FLAGS


class CaptionExtractor:
    def __init__(self):
        logging.info("New 'CaptionExtractor' instance has been initialized.")

        # Variables for computing metrics and performing transformations
        self.stemmer = nltk.stem.WordNetLemmatizer()
        self.vectorizer = CountVectorizer()

        # Variables related to assisting in the generating guidance captions
        self.captions = helpers.get_data('captions')
        self.cider = Cider(n=FLAGS.ngrams)

        # ETL
        if len(self.captions.keys()) == 0:
            self.annotations_data, self.images_data = self.get_annotations()
            self.make_caption_representations()

            # Save the dictionary for future use
            helpers.save_obj(self.captions, 'captions')

    @staticmethod
    def clean_sentence(a):
        return re.sub(r'[^\w\s]', '', a)

    @staticmethod
    def get_annotations(path=helpers.get_captions_path()):
        with open(path) as data_file:
            data = json.load(data_file)
            return data['annotations'], data['images']

    def get_guidance_caption(self, nearest_neighbors, inference=False):
        """
        Return the guidance caption for each example in a batch

        :param nearest_neighbors:
        :param inference: whether or not this is for inference (vs training)
        :return: guidance caption for each example of shape [batch size, 1]
        """

        guidance_caption = [None] * FLAGS.batch_size
        lock = Lock()

        def stem(extractor, caption):
            caption = extractor.tokenize_sentence(caption)
            caption = ' '.join(caption)
            caption = extractor.clean_sentence(caption)
            return caption

        # Update batch guidance caption list given an example's candidate captions
        def update_guidance_caption(extractor, neighbors, index):
            # Filter full captions list to get captions relevant to our neighbors
            captions = {k: v for k, v in extractor.captions.items() if k in neighbors}

            # Stem words in our captions
            gts = {}
            for image, captions in captions:
                gts[image] = [stem(extractor, caption) for caption in captions]

            candidates = [list(itertools.chain(*captions)) for captions in gts.values()]
            candidates = [stem(extractor, candidate) for candidate in candidates]

            # Dictionary with key <image> and value <tokenized reference sentence>
            res = {example_index: [caption] for example_index, caption in enumerate(candidates)}

            score, scores = extractor.cider.compute_score(gts, res)
            print(score)

            if inference:
                # Select the highest scoring caption
                score_index = scores.index(max(scores))
                guidance = candidates[score_index]
            else:
                # Select a random caption from the top k to prevent overfitting during learning
                indices = np.argpartition(candidates, -FLAGS.k)[-FLAGS.k:]
                top_captions = candidates[indices]
                guidance = top_captions[random.randint(FLAGS.k - 1)]

            with lock:
                guidance_caption[index] = guidance

        # Iterate through each example's candidate captions and select the appropriate guidance caption
        threads = []
        for i, n in enumerate(nearest_neighbors):
            t = Thread(target=update_guidance_caption, args=(self, n, i, ))
            t.start()
            threads.append(t)

        [t.join() for t in threads]
        return guidance_caption

    def make_caption_representations(self):
        """
        Creates the caption representation in the form of a list of ngrams and populates the term frequency record
        """

        # Iterature through the annotations data and find all captions belonging to our image
        for image in self.images_data:
            for annotation in self.annotations_data:
                filename = image['file_name']
                image_id = image['id']

                if annotation['image_id'] == image_id:
                    if filename not in self.captions:
                        self.captions[filename] = []

                    self.captions[filename].append(annotation['caption'])

    def stem_word(self, word):
        return self.stemmer.lemmatize(word.lower())

    def tokenize_sentence(self, sentence):
        return [self.stem_word(word) for word in nltk.word_tokenize(sentence)]

    def tokenize_sentences(self, sentences):
        tokenized_sentences = []
        for sentence in sentences:
            ts = self.tokenize_sentence(sentence)
            tokenized_sentences.append(ts)
        return tokenized_sentences
