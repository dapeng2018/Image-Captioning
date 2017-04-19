"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

from cider.cider import Cider
import helpers
import json
import logging
import nltk
import numpy as np
import random
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

FLAGS = tf.flags.FLAGS


class CaptionExtractor:
    def __init__(self, candidate_captions):
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

        self.guidance_caption_train = candidate_captions[random.randint(0, FLAGS.k)]

    @staticmethod
    def get_annotations(path=helpers.get_captions_path()):
        with open(path) as data_file:
            data = json.load(data_file)
            return data['annotations'], data['images']

    def get_guidance_caption(self, candidate_captions, inference=False):
        gts, res = {}, {}

        for i, caption in enumerate(candidate_captions):
            res[i] = self.tokenize_sentence(caption)
            gts[i] = [self.tokenize_sentence(reference)
                      for reference in (set(candidate_captions) - set([caption]))]

        score, scores = self.cider.compute_score(gts, res)

        if inference:
            # Select the highest scoring caption
            index = scores.index(max(scores))
            return candidate_captions[index]
        else:
            # Select a random caption from the top k
            indices = np.argpartition(candidate_captions, -FLAGS.k)[-FLAGS.k:]
            top_captions = candidate_captions[indices]
            return top_captions[random.randint(FLAGS.k - 1)]

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
