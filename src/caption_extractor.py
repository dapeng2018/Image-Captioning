"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import itertools
import json
import tensorflow as tf
import math
import nltk
from neighbor import Neighbor
from sklearn.feature_extraction.text import CountVectorizer

NUM_GRAMS = 4
NUM_NEIGHBORS = 60


class CaptionExtractor:
    def __init__(self):
        print("New 'CaptionExtractor' instance has been initialized.")

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

    def build(self, input_placeholder):
        nearest_neighbors = self.neighbor.nearest(input_placeholder)
        candidate_dict = {k: self.captions[k] for k in nearest_neighbors.keys()}

        candidate_captions = []
        for k, v in candidate_dict.items():
            for c in v:
                candidate_captions.append(c)
        candidate_key_permutations = list(itertools.permutations(candidate_dict.keys()))

        consensus_scores = tf.map_fn(self.get_consensus_score, candidate_key_permutations)
        self.consensus_caption = tf.argmax(consensus_scores)

    '''
    ETL related functions
    '''

    @staticmethod
    def get_annotations(path=helpers.get_captions_path()):
        """

        :param path:
        :return:
        """

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
        """
        Given a caption, this returns its apprioriate n-gram representation and updates the gram frequencies list

        :param filename
        :param image_id:
        :param lock:
        :param annotation:
        :return: gram_caption
        """

        if annotation['image_id'] == image_id:
            if filename not in self.captions:
                self.captions[filename] = {}

            caption = annotation['caption']
            stemmed_caption = self.stem_sentence(caption)
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
        for image in self.images_data[:1]:
            for annotation in self.annotations_data:
                self.make_caption_representation(image['file_name'], image['id'], annotation)

    def stem_sentence(self, sentence):
        return [self.stem_word(word) for word in nltk.word_tokenize(sentence)]

    def stem_word(self, word):
        return self.stemmer.lemmatize(word.lower())

    '''
    Consensus-based Image Description Evaluation (CIDEr) related functions
    '''

    def get_cider_score(self, candidate, descriptions):
        """

        :param candidate:
        :param descriptions:
        :return:
        """

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
        """

        :param candidate:
        :param descriptions:
        :return:
        """

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
        """

        :param gram:
        :param reference_sentence:
        :return:
        """

        term_frequency = self.get_term_reference_frequency(gram, reference_sentence)
        doc_frequency = self.get_term_frequency(gram)
        frequency = term_frequency / doc_frequency

        doc_size = len(self.images_data.keys())
        reference_occurence = self.get_doc_frequency(gram)
        rarity = math.log(doc_size / reference_occurence)

        tfidf = frequency * rarity
        return tfidf
