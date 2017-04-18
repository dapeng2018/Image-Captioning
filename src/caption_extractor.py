"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

from cider.cider import Cider
import helpers
import json
import logging
import nltk
import random
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

FLAGS = tf.flags.FLAGS


class CaptionExtractor:
    def __init__(self, candidate_captions, is_training=True):
        logging.info("New 'CaptionExtractor' instance has been initialized.")

        # Variables for computing metrics and performing transformations
        self.stemmer = nltk.stem.WordNetLemmatizer()
        self.vectorizer = CountVectorizer()

        # Variables related to assisting in the generating guidance captions
        self.gram_frequencies = helpers.get_data('gram_frequencies')
        self.captions = helpers.get_data('captions')

        # ETL
        if len(self.gram_frequencies.keys()) == 0 or len(self.captions.keys()) == 0:
            self.annotations_data, self.images_data = self.get_annotations()
            self.make_caption_representations()

            # Save the dictionaries for future use
            helpers.save_obj(self.gram_frequencies, 'gram_frequencies')
            helpers.save_obj(self.captions, 'captions')

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

    @staticmethod
    def get_annotations(path=helpers.get_captions_path()):
        with open(path) as data_file:
            data = json.load(data_file)
            return data['annotations'], data['images']

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
