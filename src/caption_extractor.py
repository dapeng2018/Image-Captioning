"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import itertools
import json
import logging
import math
import nltk
import random
import tensorflow as tf
from neighbor import Neighbor
from sklearn.feature_extraction.text import CountVectorizer

FLAGS = tf.flags.FLAGS
NUM_GRAMS = 4
NUM_NEIGHBORS = 60


class CaptionExtractor:
    def __init__(self, candidate_captions):
        print("New 'CaptionExtractor' instance has been initialized.")

        '''
        For training
        '''

        self.guidance_caption_train = candidate_captions[random.randint(0, FLAGS.k)]

        '''
        For inference
        '''

        # Variables for computing metrics and performing transformations
        self.neighbor = Neighbor()
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


        """""
        # Compute the consensus score and find the highest scoring caption
        test_candidate_captions = 0
        caption_combos = itertools.combinations(test_candidate_captions, 2)
        consensus_scores = tf.map_fn(self.get_consensus_score, caption_combos)

        indices = tf.argmax(consensus_scores, dimension=1)
        self.guidance_caption_test = tf.gather(test_candidate_captions, indices)
        """

    '''
    ETL related functions
    '''

    @staticmethod
    def extend_to_len(x, n=300):
        x.extend(['' for _ in range(n - len(x))])

    @staticmethod
    def get_annotations(path=helpers.get_captions_path()):
        with open(path) as data_file:
            data = json.load(data_file)
            return data['annotations'], data['images']

    def get_consensus_score(self, caption, collection):
        """

        :param caption:
        :param collection:
        :return:
        """

        score = 0
        for c in collection:
            score += self.get_cider_scores(caption, [c])
        score /= (len(collection) - 1)

        return score

    @staticmethod
    def get_gram_representation(word_list, n=NUM_GRAMS):
        return list(nltk.everygrams(word_list, min_len=1, max_len=n))

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

    def tokenize_sentence(self, sentence):
        return [self.stem_word(word) for word in nltk.word_tokenize(sentence)]

    def stem_word(self, word):
        return self.stemmer.lemmatize(word.lower())

    '''
    Consensus-based Image Description Evaluation (CIDEr) related functions
    '''

    def get_cider_score(self, candidate, descriptions):
        num_descriptions = len(descriptions)
        score = 0

        for description in descriptions:
            # Measure term values of candidate and description sentences using TF-IDF
            tfidf_candidate = self.get_tfidf(candidate)
            tfidf_description = self.get_tfidf(description)

            # Compute the CIDEr score by getting the average of their cosine similarities
            cosine_similarities = tf.convert_to_tensor(helpers.get_cosine_similarity(tfidf_candidate, tfidf_description))
            score += cosine_similarities

        score /= num_descriptions
        return score

    def get_cider_scores(self, candidate, descriptions):
        score = 0

        for i in range(NUM_GRAMS):
            candidate = self.get_grams_of_size(i, candidate)
            descriptions = [self.get_grams_of_size(i, description) for description in descriptions]
            score += self.get_cider_score(candidate, descriptions)

        return score

    def get_doc_frequency(self, gram):
        return self.gram_frequencies[str(gram)]['image_count']

    @staticmethod
    def get_grams_of_size(n, representation):
        return filter(representation, lambda x: len(x) == n)

    def get_similarity(self, caption, captions):
        pass

    def get_term_frequency(self, gram):
        return self.gram_frequencies[str(gram)]['count']

    @staticmethod
    def get_term_frequency(gram, reference_senence):
        return reference_senence.count(gram)

    def get_tfidf(self, gram, reference_sentence):
        term_frequency = self.get_term_reference_frequency(gram, reference_sentence)
        doc_frequency = self.get_term_frequency(gram)
        frequency = term_frequency / doc_frequency

        doc_size = len(self.images_data.keys())
        reference_occurence = self.get_doc_frequency(gram)
        rarity = math.log(doc_size / reference_occurence)

        tfidf = frequency * rarity
        return tfidf
