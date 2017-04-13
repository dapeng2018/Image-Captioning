"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import json
import tensorflow as tf
import nltk
from sklearn.feature_extraction.text import CountVectorizer


class Caption:
    def __init__(self):
        self.annotations_data, self.images_data = self.get_annotations()
        self.lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
        self.vectorizer = CountVectorizer()

    def extract_guidance(self):
        pass

    @staticmethod
    def get_annotations(path=helpers.get_captions_path()):
        with open(path) as data_file:
            data = json.load(data_file)
            return data['annotations'], data['images']

    def get_consenses_score(self, caption, descriptions):
        # Stem our sentences and get their n-gram representations
        caption = self.get_gram_representation(caption)
        descriptions = [self.get_gram_representation(c) for c in descriptions]

        # Compute the consensus score by averaging similarity to all other captions
        total_similarity = tf.reduce_sum(self.get_similarity(caption, descriptions))
        size = len(descriptions) - 1
        score = total_similarity / size

        return score

    @staticmethod
    def get_gram_representation(sentence, n=4):
        return nltk.ngrams(sentence.split(), n)

    def stem_sentence(self, sentence):
        return [self.stem_sentence(word) for word in sentence]

    def stem_word(self, word):
        return self.lmtzr.lemmatize(word)

    '''
    Consensus-based Image Description Evaluation (CIDEr) related functions
    '''

    @staticmethod
    def get_cosine_similarity(a, b):
        return (a * b) / (tf.abs(a) * tf.abs(b))

    def get_cider_score(self, candidate, description):
        # Measure term values of candidate and description sentences using TF-IDF
        tfidf_candidate = self.get_tfidf(candidate)
        tfidf_description = self.get_tfidf(description)

        # Compute the CIDEr score by getting the average of their cosine similarities
        cosine_similarities = self.get_cosine_similarity(tfidf_candidate, tfidf_description)
        score = tf.reduce_mean(cosine_similarities)

        return score

    def get_cider_scores(self):
        pass

    def get_similarity(self, caption, captions):
        pass

    def get_doc_frequency(self):
        pass

    def get_term_fruency(self):
        pass

    def get_tfidf(self, gram_represenation):
        return 0
