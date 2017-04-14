"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import json
import tensorflow as tf
import math
import nltk
from functools import partial
from neighbor import Neighbor
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing.dummy import Pool as ThreadPool, Lock

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
        self.gram_frequencies = {}
        self.annotations_data, self.images_data = self.get_annotations()
        self.captions = {}

        # ETL
        self.make_caption_representations()

    def build(self, input_placeholder):
        nearest_neighbors = self.neighbor.nearest(input_placeholder)
        candidate_captions = {k: self.captions[k] if k in self.captions else next for k in nearest_neighbors.keys()}
        consensus_scores = tf.foldl(self.get_consensus_score())
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

    def get_caption_representation(self, filename, image_id, lock, annotation):
        """
        Given a caption, this returns its apprioriate n-gram representation and updates the gram frequencies list

        :param filename
        :param image_id:
        :param lock:
        :param annotation:
        :return: gram_caption
        """

        if annotation['image_id'] == image_id:
            caption = annotation['caption']
            stemmed_caption = self.stem_sentence(caption)
            gram_caption = self.get_gram_representation(stemmed_caption)

            used = []
            for gram in gram_caption:
                if str(gram) in self.gram_frequencies:
                    lock.acquire()
                    self.gram_frequencies[str(gram)]['count'] += 1
                    if gram not in used:
                        self.gram_frequencies[str(gram)]['image_count'] += 1
                else:
                    lock.acquire()
                    self.gram_frequencies[str(gram)] = {'count': 1, 'image_count': 1}

                used.append(gram)
                lock.release()

            self.captions[filename] = gram_caption

    def make_caption_representations(self):
        """
        Creates the caption representation in the form of a list of ngrams and populates the term frequency record
        """

        for image in self.images_data[:1]:
            # Reference values pertaining a particular image
            filename = image['file_name']
            image_id = image['id']

            # Iterature through the annotations data and find all captions belonging to our image
            pool = ThreadPool(4)
            lock = Lock()
            get_caption_partial = partial(self.get_caption_representation, filename, image_id, lock)
            pool.map(get_caption_partial, self.annotations_data)

    def make_gram_frequencies(self):
        """

        :return: frequencies
        """

        frequencies = {}
        for _, captions in self.captions.items():
            for caption in captions:
                for gram in caption:
                    g = str(gram)
                    if g in frequencies:
                        frequencies[g] += 1
                    else:
                        frequencies[g] = 1
        return frequencies

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
