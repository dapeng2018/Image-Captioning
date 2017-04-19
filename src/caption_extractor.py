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
        self.gram_frequencies = helpers.get_data('gram_frequencies')

        # ETL
        if len(self.gram_frequencies.keys()) == 0 or len(self.captions.keys()) == 0:
            self.annotations_data, self.images_data = self.get_annotations()
            self.make_caption_representations()

            # Save the dictionaries for future use
            helpers.save_obj(self.gram_frequencies, 'gram_frequencies')
            helpers.save_obj(self.captions, 'captions')

    @staticmethod
    def clean_sentence(a):
        return re.sub(r'[^\w\s]', '', a)

    @staticmethod
    def get_annotations(path=helpers.get_captions_path()):
        with open(path) as data_file:
            data = json.load(data_file)
            return data['annotations'], data['images']

    def get_candidates_from_neighbors(self, nearest_neighbors):
        """
        Return a set of candidate captions for each image giventhe nearest nearest neighbors

        :param nearest_neighbors: list of filenames of shape [batch size, FLAGS.n]
        :return: candidate captions of shape [batch size, FLAGS.k]
        """

        candidate_captions = [None] * FLAGS.batch_size
        lock = Lock()

        # Update batch candidate captions list given an example's nearest neighbors
        def update_candidate_captions(extractor, neighbors, index):

            # Get the captions from the captions list given a decoded basename of the paths
            captions = [extractor.captions[os.path.basename(filename.decode('UTF-8'))]
                        for filename in neighbors]
            captions = list(itertools.chain(*captions))

            with lock:
                candidate_captions[index] = captions

        # Iterate through each example's nearest neighbors in the batch
        threads = []
        for i, n in enumerate(nearest_neighbors):
            t = Thread(target=update_candidate_captions, args=(self, n, i, ))
            t.start()
            threads.append(t)

        [t.join() for t in threads]
        return candidate_captions

    def get_guidance_caption(self, candidate_captions, inference=False):
        """
        Return the guidance caption for each example in a batch

        :param candidate_captions: list of candidates for each example of shape [batch size, FLAGS.k]
        :param inference: whether or not this is for inference (vs training)
        :return: guidance caption for each example of shape [batch size, 1]
        """

        guidance_caption = [None] * FLAGS.batch_size
        lock = Lock()

        # Update batch guidance caption list given an example's candidate captions
        def update_guidance_caption(extractor, candidates, index):
            gts = {}  # dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
            res = {}  # dictionary with key <image> and value <tokenized reference sentence>

            for example_index, caption in enumerate(candidates):
                caption = extractor.clean_sentence(caption)
                gts[example_index] = [extractor.tokenizer.tokenize(reference)
                                      for reference in (set(candidates) - set([caption]))]
                res[example_index] = extractor.tokenizer.tokenize(caption)

            score, scores = extractor.cider.compute_score(gts, res)

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
        for i, c in enumerate(candidate_captions):
            t = Thread(target=update_guidance_caption, args=(self, c, i, ))
            t.start()
            threads.append(t)

        [t.join() for t in threads]
        return guidance_caption

    def make_caption_representation(self, filename, image_id, annotation):
        if annotation['image_id'] == image_id:
            if filename not in self.captions:
                self.captions[filename] = {}

            caption = annotation['caption']
            stemmed_caption = self.tokenize_sentence(caption)
            self.captions[filename][caption] = self.get_gram_representation(stemmed_caption)

            used = []
            for gram in self.captions[filename][caption]:
                if str(gram) in self.gram_frequencies:
                    self.gram_frequencies[str(gram)]['count'] += 1
                    if gram not in used:
                        self.gram_frequencies[str(gram)]['image_count'] += 1
                        used.append(gram)
                else:
                    self.gram_frequencies[str(gram)] = {'count': 1, 'image_count': 1}
                    used.append(gram)

    def make_caption_representations(self):
        """
        Creates the caption representation in the form of a list of ngrams and populates the term frequency record
        """

        # Iterature through the annotations data and find all captions belonging to our image
        for image in self.images_data:
            for annotation in self.annotations_data:
                self.make_caption_representation(image['file_name'], image['id'], annotation)

    def stem_word(self, word):
        return self.stemmer.lemmatize(word.lower())

    def tokenize_sentence(self, sentence):
        return [self.stem_word(word) for word in nltk.word_tokenize(sentence)]
