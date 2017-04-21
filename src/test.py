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
    candidate_captions_ph = tf.placeholder(dtype=tf.string, shape=[None, FLAGS.n * 5])
    caption_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.stv_size])
    image_conv_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, k, k, FLAGS.conv_size])
    image_fc_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, k, k, 4096])
    image_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.train_height, FLAGS.train_width, 3])
    rnn_inputs_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, 5])
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
    tatt = Attention(image_conv_encoding_ph, caption_encoding_ph)
    decoder = Decoder(tatt.context_vector, rnn_inputs_ph)

    # Retrieve training images for caption extraction
    example_image, example_filename = helpers.next_example(height=FLAGS.train_height, width=FLAGS.train_width)
    all_examples, all_filenames = tf.train.batch([example_image, example_filename],
                                                 helpers.get_training_size(),
                                                 num_threads=8,
                                                 capacity=10000)

    # Initialize session and begin threads
    sess.run(tf.global_variables_initializer())
    logging.info("Begining training..")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()

    # Restore previously trained model
    saved_path = os.path.abspath(FLAGS.model_path)
    saver = tf.train.Saver()
    #saver.restore(sess, saved_path)

    # Evaluate training images and imag encodings
    all_examples_eval = all_examples.eval()
    all_filenames_eval = all_filenames.eval()
    input_fc_encoding = fc_encoding.eval(feed_dict={image_ph: input_image})
    training_fc_encodings = fc_encoding.eval(feed_dict={image_ph: all_examples_eval})

    # Get nearest neighbor images to get list of candidate captions
    neighbor_dict = {
        image_fc_encoding_ph: input_fc_encoding,
        training_fc_encodings_ph: training_fc_encodings,
        training_filenames_ph: all_filenames_eval}
    nearest_neighbors = neighbor.nearest.eval(feed_dict=neighbor_dict)

    # Get candidate captions
    extractor = CaptionExtractor()

    # Extract guidance caption as the top CIDEr scoring sentence
    guidance_captions = extractor.get_guidance_caption(nearest_neighbors, inference=False)

    # Compute context vector using the guidance caption and image encodings
    guidance_caption_encoding = stv.encode(guidance_captions, batch_size=1, use_eos=True)
    context_vector = tatt.context_vector

    # Set up ops and vars for decoding the caption
    input_conv_encoding = conv_encoding.eval(feed_dict={image_ph: input_image})
    predicted_index = tf.argmax(decoder.output, axis=1)
    predicted_word = tf.gather(vocab.list, predicted_index)

    rnn_inputs = vocab.get_bos_rnn_input()
    feed_dict = {caption_encoding_ph: guidance_caption_encoding,
                 image_conv_encoding_ph: input_conv_encoding,
                 rnn_inputs_ph: np.array(rnn_inputs)}

    # Decode caption
    caption = []

    for _ in range(FLAGS.max_caption_size):
        # Evaludate the literal string from the prediction
        word, word_index = sess.run([predicted_word, predicted_index], feed_dict=feed_dict)

        # Since this is not for a batch, get the first elements
        word = word[0].decode('UTF-8')
        word_index = word_index[0]

        # If the prediction was <eos>, break the loop
        if word == '<eos>':
            break

        # Append word to the running caption
        caption.append(word)

        # Make the next input for the decoder
        predicted_1hot = helpers.index_to_1hot(word_index)
        predicted_1hot = [[predicted_1hot]]
        rnn_inputs = np.concatenate((rnn_inputs, predicted_1hot), axis=1)
        feed_dict[rnn_inputs_ph] = np.array(rnn_inputs)

    # Convert caption array into string and print it
    caption = ' '.join(caption)
    logging.info("OUTPUT: %s" % caption)

    # Join threads
    coord.request_stop()
    coord.join(threads)
