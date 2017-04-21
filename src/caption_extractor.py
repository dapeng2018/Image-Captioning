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
from multiprocessing import Lock, Manager, Process

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
    def clean_sentence(sentence):
        return re.sub(r'[^\w\s]', '', sentence)

    @staticmethod
    def extend_to_len(x, n=512):
        x.extend(['' for _ in range(n - len(x))])

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

        def stem(extractor, caption):
            caption = extractor.tokenize_sentence(caption)
            caption = ' '.join(caption)
            caption = extractor.clean_sentence(caption)
            return caption

        # Iterate through each example's candidate captions and select the appropriate guidance caption
        for index, neighbors in enumerate(nearest_neighbors):
            # Filter full captions list to get captions relevant to our neighbors
            neighbors = [os.path.basename(neighbor.decode('UTF-8')) for neighbor in neighbors]
            captions = {k: v for k, v in self.captions.items() if k in neighbors}

            # Flatten candidate captions into one list and stem all their words
            candidates = list(itertools.chain(*captions.values()))
            candidates = [stem(self, candidate) for candidate in candidates]

            # Compute CIDEr scores through in parallel
            with Manager() as manager:
                total_scores = manager.dict()
                threads = []
                lock = Lock()

                def update_scores(c):
                    ref = {filename: [c] for filename in captions.keys()}
                    score, _ = self.cider.compute_score(captions, ref)

                    with lock:
                        total_scores[c] = score

                for candidate in candidates:
                    t = Process(target=update_scores, args=(candidate, ))
                    t.start()
                    threads.append(t)

                [t.join() for t in threads]
                scores = [value for value in total_scores.values()]

            if inference:
                # Select the highest scoring caption
                score_index = scores.index(max(scores))
                guidance = candidates[score_index]
            else:
                # Select a random caption from the top k to prevent over-fitting during learning
                k = FLAGS.k if len(scores) >= FLAGS.k else len(scores)
                indices = np.argpartition(scores, -k)[-k:]
                top_captions = [candidates[top_index] for top_index in indices]
                guidance = top_captions[random.randint(0, k - 1)]

            guidance_caption[index] = guidance

        return guidance_caption

    def make_caption_representations(self):
        """
        Creates the caption representation in the form of a list of ngrams and populates the term frequency record
        """

        # Iterate through the annotations data and find all captions belonging to our image
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

    def tokenize_sentences(self, sentences, extend=False):
        tokenized_sentences = []

        for sentence in sentences:
            ts = self.tokenize_sentence(sentence)
            if extend:
                self.extend_to_len(ts, FLAGS.state_size)

            tokenized_sentences.append(ts)

        return tokenized_sentences
