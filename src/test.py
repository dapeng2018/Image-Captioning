"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Executable script for generating a caption given an input image and an already trained model.
"""

import logging
import helpers
import os
import stv.configuration as stv_configuration
import tensorflow as tf
from attention import Attention
from caption_extractor import CaptionExtractor
from decoder import Decoder
from neighbor import Neighbor
from stv.encoder_manager import EncoderManager
from vgg.fcn16_vgg import FCN16VGG as Vgg16
from vocab import Vocab


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('input', None, '')
tf.flags.DEFINE_string('model_path', None, '')
helpers.config_model_flags()
helpers.config_logging(env='testing')


with tf.Session as sess:
    # Init
    vocab = Vocab()

    # Initialize placeholders
    candidate_captions_ph = tf.placeholder(dtype=tf.string, shape=[FLAGS.n * 5])
    caption_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[1, FLAGS.stv_size])
    image_fc_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[1, 7, 7, 4096])
    image_ph = tf.placeholder(dtype=tf.float32, shape=[1, FLAGS.train_height, FLAGS.train_width, 3])
    training_fc_encodings_ph = tf.placeholder(dtype=tf.float32, shape=[helpers.get_training_size(), 7, 7, 4096])
    training_filenames_ph = tf.placeholder(dtype=tf.string, shape=[helpers.get_training_size()])

    # Initialize auxiliary
    image_shape = [1, FLAGS.train_height, FLAGS.train_width, 3]
    neighbor = Neighbor(image_fc_encoding_ph, training_fc_encodings_ph, training_filenames_ph)

    # Initialize skip-thought-vector model
    stv = EncoderManager()
    stv_uni_config = stv_configuration.model_config()
    stv.load_model(stv_uni_config, FLAGS.stv_vocab_file, FLAGS.stv_embeddings_file, FLAGS.stv_checkpoint_path)

    # Initialize encoders
    vgg = Vgg16()
    vgg.build(image_ph, image_shape[1:])
    conv_encoding = vgg.pool5
    fc_encoding = vgg.fc7
    extractor = CaptionExtractor(candidate_captions_ph)

    # Attention model and decoder
    tatt = Attention(conv_encoding, caption_encoding_ph)
    decoder = Decoder(tatt.context_vector)

    # Initialize sessuib and restore previously trained model
    sess.run(tf.global_variables_initializer())
    saved_path = os.path.abspath(FLAGS.model_path)
    saver = tf.train.Saver()
    saver.restore(sess, saved_path)

    # Generate caption

    sess.close()

exit(0)
