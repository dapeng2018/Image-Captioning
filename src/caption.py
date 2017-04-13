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
        # Variables for performing transformations
        self.stemmer = nltk.stem.WordNetLemmatizer()
        self.vectorizer = CountVectorizer()

        # Variables related to assisting in the generating guidance captions
        self.annotations_data, self.images_data = self.get_annotations()
        self.captions = self.make_caption_representations()
        self.gram_frequencies = self.make_gram_frequencies()

    '''
    ETL related functions
    '''

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
    def get_gram_representation(word_list, n=4):
        return list(nltk.everygrams(word_list, min_len=1, max_len=n))

    def make_caption_representations(self):
        representations = {}
        for image in self.images_data[:10]:
            # Reference values pertaining a particular image
            file_name = image['file_name']
            image_id = image['id']

            # Iterature through the annotations data and find all captions belonging to our image
            captions = []
            for annotation in self.annotations_data:
                if annotation['image_id'] == image_id:
                    caption = annotation['caption']
                    stemmed_caption = self.stem_sentence(caption)
                    gram_caption = self.get_gram_representation(stemmed_caption)
                    captions.append(gram_caption)
            representations[file_name] = captions

        return representations

    def make_gram_frequencies(self):
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

    def get_doc_frequency(self, gram):
        return self.gram_frequencies[str(gram)]

    @staticmethod
    def get_term_frequency(gram, reference_senence):
        return reference_senence.count(gram)

    def get_tfidf(self, gram, reference_sentence):
        term_frequency = self.get_term_frequency(gram, reference_sentence)
        doc_frequency = self.get_doc_frequency(gram)
        frequency = term_frequency / doc_frequency

        rarity = tf.log()

        tfidf = frequency * rarity
        return tfidf