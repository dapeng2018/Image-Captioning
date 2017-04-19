"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Executable script for generating a caption given an input image and an already trained model.
"""

import logging
import helpers
import itertools
import math
import numpy as np
import os
import stv.configuration as stv_configuration
import tensorflow as tf
import time
from attention import Attention
from caption_extractor import CaptionExtractor
from decoder import Decoder
from neighbor import Neighbor
from stv.encoder_manager import EncoderManager
from vgg.fcn16_vgg import FCN16VGG as Vgg16
from vocab import Vocab


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 1, '')
tf.flags.DEFINE_string('input', None, '')
tf.flags.DEFINE_string('model_path', None, '')
helpers.config_model_flags()
helpers.config_logging(env='testing')

if not FLAGS.input or not FLAGS.model_path:
    logging.error('You did not provide an input image path and model path.')
    exit(1)


config = helpers.get_session_config()
with tf.Session(config=config) as sess:
    # Init
    vocab = Vocab()
    input_path = os.path.abspath(FLAGS.input)
    input_image = helpers.load_image_to(input_path, height=512, width=512)
    input_image = np.reshape(input_image, [1, 512, 512, 3])
    k = math.sqrt(FLAGS.kk)

    # Initialize placeholders
    candidate_captions_ph = tf.placeholder(dtype=tf.string, shape=[1, FLAGS.n * 5])
    caption_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.stv_size])
    image_fc_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, k, k, 4096])
    image_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.train_height, FLAGS.train_width, 3])
    training_fc_encodings_ph = tf.placeholder(dtype=tf.float32, shape=[None, k, k, 4096])
    training_filenames_ph = tf.placeholder(dtype=tf.string, shape=[helpers.get_training_size()])

    # Initialize auxiliary
    image_shape = [1, FLAGS.train_height, FLAGS.train_width, 3]
    neighbor = Neighbor(image_fc_encoding_ph, training_fc_encodings_ph, training_filenames_ph)

    # Initialize skip-thought-vector model
    stv = EncoderManager()
    stv_uni_config = stv_configuration.model_config()
    stv.load_model(stv_uni_config, FLAGS.stv_vocab_file, FLAGS.stv_embeddings_file, FLAGS.stv_checkpoint_path)

    # Initialize models
    vgg = Vgg16()
    vgg.build(image_ph, image_shape[1:])
    conv_encoding = vgg.pool5
    fc_encoding = vgg.fc7
    tatt = Attention(conv_encoding, caption_encoding_ph)
    #decoder = Decoder(tatt.context_vector)

    # Retrieve training images for caption extraction
    example_image, example_filename = helpers.next_example(height=FLAGS.train_height, width=FLAGS.train_width)
    all_examples, all_filenames = tf.train.batch([example_image, example_filename],
                                                 helpers.get_training_size(),
                                                 num_threads=8,
                                                 capacity=100)

    # Initialize session and begin threads
    sess.run(tf.global_variables_initializer())
    logging.info("Begining training..")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()

    # Restore previously trained model
    saved_path = os.path.abspath(FLAGS.model_path)
    #saver = tf.train.Saver()
    #saver.restore(sess, saved_path)

    # Get nearest neighbor images to get list of candidate captions
    all_examples_eval = all_examples.eval()
    all_filenames_eval = all_filenames.eval()
    input_fc_encoding = fc_encoding.eval(feed_dict={image_ph: input_image})
    training_fc_encodings = fc_encoding.eval(feed_dict={image_ph: all_examples_eval})

    neighbor_dict = {
        image_fc_encoding_ph: input_fc_encoding,
        training_fc_encodings_ph: training_fc_encodings,
        training_filenames_ph: all_filenames_eval}
    nearest_neighbors = neighbor.nearest.eval(feed_dict=neighbor_dict)

    # Get candidate captions
    extractor = CaptionExtractor()
    candidate_captions = extractor.get_candidates_from_neighbors(nearest_neighbors)

    # Extract guidance caption as the top CIDEr scoring sentence
    guidance_caption = extractor.get_guidance_caption(candidate_captions, inference=True)

    # Compute context vector using the guidance caption and image encodings
    tokenized_caption = extractor.tokenize_sentence(guidance_caption)
    guidance_caption_encoding = stv.encode(tokenized_caption, batch_size=1)
    context_vector = tatt.context_vector

    # Decode caption
    words = decoder.get_caption(vocab)
    feed_dict = {caption_encoding_ph: guidance_caption_encoding, image_ph: input_image}
    word_list = words.eval(feed_dict=feed_dict)
    caption = ' '.join(word_list)
    print(caption)

    # Stop threads and close the tensorflow sessions
    coord.request_stop()
    coord.join(threads)
    stv.close()
    sess.close()

exit(0)
