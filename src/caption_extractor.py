"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Class responsible extracting a guidance caption and other operations related to ETL of sentences.
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
    # Remove anything that is not a character or space in a sentence (list of words)
    def clean_sentence(sentence):
        return re.sub(r'[^\w\s]', '', sentence)

    @staticmethod
    # Retrieve the MSCOCO annotations and images data in the form of dictionaries
    def get_annotations(path=helpers.get_captions_path()):
        with open(path) as data_file:
            data = json.load(data_file)
            return data['annotations'], data['images']

    @staticmethod
    # Make a word lowercased and stem it
    def stem_word(stemmer, word):
        return stemmer.lemmatize(word.lower())

    def get_guidance_caption(self, nearest_neighbors, inference=False):
        """
        Return the guidance caption for each example in a batch

        :param nearest_neighbors: set of nearest neighbors for a batch of images [batch size, FLAGS.n]
        :param inference: whether or not this is for inference (vs training)
        :return: guidance caption for each example of shape [batch size, 1]
        """

        with Manager() as manager:
            guidance_caption = manager.list(range(FLAGS.batch_size))

            def stem(extractor, stemmer, caption):
                caption = extractor.tokenize_sentence(stemmer, caption)
                caption = ' '.join(caption)
                caption = extractor.clean_sentence(caption)
                return caption

            # Iterate through each example's candidate captions and select the appropriate guidance caption
            def get_example_guidance(neighbors, index):
                stemmer = nltk.stem.WordNetLemmatizer()

                # Filter full captions list to get captions relevant to our neighbors
                neighbors = [os.path.basename(neighbor.decode('UTF-8')) for neighbor in neighbors]
                captions = {k: v for k, v in self.captions.items() if k in neighbors}

                # Flatten candidate captions into one list and stem all their words
                candidates = list(itertools.chain(*captions.values()))
                candidates = [stem(self, stemmer, candidate) for candidate in candidates]

                # Compute CIDEr scores in parallel
                with Manager() as cider_manager:
                    total_scores = cider_manager.dict()
                    cider_threads = []
                    cider_lock = Lock()

                    def update_scores(c):
                        ref = {filename: [c] for filename in captions.keys()}
                        score, _ = self.cider.compute_score(captions, ref)

                        with cider_lock:
                            total_scores[c] = score

                    for candidate in candidates:
                        ct = Process(target=update_scores, args=(candidate, ))
                        ct.start()
                        cider_threads.append(ct)

                    [ct.join() for ct in cider_threads]
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

            threads = []
            for i, n in enumerate(nearest_neighbors):
                t = Process(target=get_example_guidance, args=(n, i, ))
                t.start()
                threads.append(t)

            [t.join() for t in threads]
            return list(guidance_caption)

    # Create a dictionary storing training image names with their associated captions
    def make_caption_representations(self):
        # Iterate through the annotations data and find all captions belonging to our image
        for image in self.images_data:
            for annotation in self.annotations_data:
                filename = image['file_name']
                image_id = image['id']

                if annotation['image_id'] == image_id:
                    if filename not in self.captions:
                        self.captions[filename] = []

                    self.captions[filename].append(annotation['caption'])

    # Tokenize a given sentence
    def tokenize_sentence(self, stemmer, sentence):
        return [self.stem_word(stemmer, word) for word in nltk.word_tokenize(sentence)]

    # Tokenize a set of sentences and pad it if specified
    def tokenize_sentences(self, sentences):
        tokenized_sentences = []

        for sentence in sentences:
            ts = self.tokenize_sentence(sentence)
            tokenized_sentences.append(ts)

        return tokenized_sentences
