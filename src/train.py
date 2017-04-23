"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Executable script for training a new captioning model.
"""

import helpers
import logging
import math
import numpy as np
import random
import tensorflow as tf
import time

import stv.configuration as stv_configuration
from attention import Attention
from caption_extractor import CaptionExtractor
from decoder import Decoder
from neighbor import Neighbor
from stv.encoder_manager import EncoderManager
from vgg.fcn16_vgg import FCN16VGG as Vgg16
from vocab import Vocab

FLAGS = tf.flags.FLAGS
helpers.config_model_flags()
helpers.config_logging()

# Optimization flags
tf.flags.DEFINE_integer('batch_size', 16, 'Mini-Batch size of images')
tf.flags.DEFINE_integer('epochs', 100, 'Number of training iterations')
tf.flags.DEFINE_float('learning_rate', 4e-4, 'Optimizer learning rate')
tf.flags.DEFINE_float('learning_rate_dec_factor', .8, 'Factor in which the learning rate decreases')
tf.flags.DEFINE_integer('learning_rate_dec_freq', 3, 'How often (iterations) the learning rate decreases')
tf.flags.DEFINE_integer('learning_rate_dec_thresh', 10, 'Number of iterations before learning rate starts decreasing')

# Scheduled sampling flags
tf.flags.DEFINE_float('sched_c', .1, 'Offset constant for decaying scheduled sampling rate in linear decay')
tf.flags.DEFINE_string('sched_dec_type', 'lin', 'Scheduled sampling rate decay type. [lin, exp, inv]')
tf.flags.DEFINE_float('sched_k', 1, 'Constant for decaying scheduled sampling rate. k < 1 in exp and k >= 1 in inv')
tf.flags.DEFINE_integer('sched_start', 10, 'When (epoch) scheduled sampling should begin')

# Misc flags
tf.flags.DEFINE_integer('print_every', 100, 'How often (iterations) to log the current progress of training')
tf.flags.DEFINE_integer('save_every', 1, 'How often epochs) to save the current state of the model')


config = helpers.get_session_config()
with tf.Session(config=config) as sess:
    # Init
    vocab = Vocab()
    k = math.sqrt(FLAGS.kk)

    # Initialize placeholders
    candidate_captions_ph = tf.placeholder(dtype=tf.string, shape=[None, FLAGS.n * 5])
    caption_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.stv_size])
    context_vector_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.conv_size])
    image_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.train_height, FLAGS.train_width, 3])
    image_conv_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, k, k, FLAGS.conv_size])
    image_fc_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, k, k, 4096])
    image_name_ph = tf.placeholder(dtype=tf.string)
    labels_ph = tf.placeholder(tf.float32, shape=(None, FLAGS.vocab_size))
    learning_rate_ph = tf.placeholder(dtype=tf.float32)
    rnn_inputs_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])
    training_fc_encodings_ph = tf.placeholder(dtype=tf.float32, shape=[helpers.get_training_size(), k, k, 4096])
    training_filenames_ph = tf.placeholder(dtype=tf.string, shape=[helpers.get_training_size()])

    # Initialize auxiliary
    image_shape = [1, FLAGS.train_height, FLAGS.train_width, 3]
    neighbor = Neighbor(image_fc_encoding_ph, training_fc_encodings_ph, training_filenames_ph)

    # Initialize encoders
    with tf.name_scope('encoders'):
        vgg = Vgg16()
        vgg.build(image_ph, image_shape[1:])
        conv_encoding = vgg.pool5
        fc_encoding = vgg.fc7

    # Initialize guidance caption extractor and skip-thought-vector model
    extractor = CaptionExtractor()
    stv = EncoderManager()
    stv_uni_config = stv_configuration.model_config()
    stv.load_model(stv_uni_config, FLAGS.stv_vocab_file, FLAGS.stv_embeddings_file, FLAGS.stv_checkpoint_path)

    # Attention model and decoder
    with tf.variable_scope('trained'):
        tatt = Attention(image_conv_encoding_ph, caption_encoding_ph)
        decoder = Decoder(context_vector_ph, rnn_inputs_ph)

    # Set up ops for decoding the caption
    predicted_index = tf.argmax(decoder.output, axis=1)
    predicted_index = tf.expand_dims(predicted_index, axis=1)
    sampled_index = decoder.sample()

    # Optimization ops
    with tf.name_scope('optimization'):
        cross_entropy = -tf.reduce_sum(labels_ph * tf.log(decoder.output + FLAGS.epsilon))
        optimizer = tf.train.AdamOptimizer(learning_rate_ph)
        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='trained')
        model_grads = optimizer.compute_gradients(cross_entropy, model_vars)
        update_step = optimizer.apply_gradients(model_grads)

    # Training data ops
    example_image, example_filename = helpers.next_example(height=FLAGS.train_height, width=FLAGS.train_width)
    capacity = FLAGS.batch_size * 2
    batch_examples, batch_filenames = tf.train.batch([example_image, example_filename],
                                                     FLAGS.batch_size,
                                                     num_threads=8,
                                                     capacity=capacity)
    all_examples, all_filenames = tf.train.batch([example_image, example_filename],
                                                 helpers.get_training_size(),
                                                 num_threads=8,
                                                 capacity=10000)

    # Initialize session and threads then begin training
    logging.info("Begining training..")
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()
    saver = tf.train.Saver(model_vars)

    # Evaluate training examples now since they do not need to recomputed in our loop
    all_examples_eval = all_examples.eval()
    all_filenames_eval = all_filenames.eval()
    training_fc_encodings = fc_encoding.eval(feed_dict={image_ph: all_examples_eval})

    # Optimization loop
    for e in range(FLAGS.epochs):
        num_iterations = math.floor(helpers.get_training_size() // FLAGS.batch_size)

        # Iterate through our entire training dataset
        for i in range(num_iterations + 1):
            # Compute image encodings
            example_images = batch_examples.eval()
            example_conv_encodings = conv_encoding.eval(feed_dict={image_ph: example_images})
            examples_fc_encoding = fc_encoding.eval(feed_dict={image_ph: example_images})

            # Get nearest neighboring images
            neighbor_dict = {
                image_fc_encoding_ph: examples_fc_encoding,
                training_fc_encodings_ph: training_fc_encodings,
                training_filenames_ph: all_filenames_eval}
            nearest_neighbors = neighbor.nearest.eval(feed_dict=neighbor_dict)

            # Get guidance caption encodings
            guidance_captions = extractor.get_guidance_caption(nearest_neighbors)
            guidance_caption_encodings = stv.encode(guidance_captions, batch_size=FLAGS.batch_size, use_eos=True)

            # Set up RNN inputs and labels
            rnn_inputs = vocab.get_bos_rnn_input(FLAGS.batch_size)
            rnn_word_labels = [vocab.add_bos_eos(extractor.tokenize_sentence(extractor.stemmer, gc))
                               for gc in guidance_captions]
            rnn_1hot_labels = vocab.word_labels_to_1hot(rnn_word_labels)
            bos_labels = np.array(rnn_1hot_labels)[:, 0].reshape((-1, FLAGS.vocab_size))

            # Compute the context vector so it is not recomputed for each time step of the LSTM
            context_dict = {caption_encoding_ph: guidance_caption_encodings,
                            image_conv_encoding_ph: example_conv_encodings}
            context_vector = tatt.context_vector.eval(feed_dict=context_dict)

            # Feed for the update
            feed_dict = {context_vector_ph: context_vector,
                         learning_rate_ph: FLAGS.learning_rate,
                         rnn_inputs_ph: rnn_inputs,
                         labels_ph: bos_labels}

            # Update weights
            for w in range(FLAGS.max_caption_size - 1):
                # Scheduled sampling
                if e >= FLAGS.sched_start and random.random() >= FLAGS.sched_rate:
                    # Use sample
                    _predicted_index = sampled_index
                else:
                    # Use ground-truth
                    _predicted_index = predicted_index

                # Update
                word_indices, loss, _ = sess.run([_predicted_index, cross_entropy, update_step], feed_dict=feed_dict)

                # Prepare next time step's inputs and labels
                rnn_inputs = np.concatenate((feed_dict[rnn_inputs_ph], word_indices), axis=1)
                feed_dict[rnn_inputs_ph] = rnn_inputs
                feed_dict[labels_ph] = np.array(rnn_1hot_labels)[:, w + 1].reshape((-1, FLAGS.vocab_size))

            # Log loss
            if i % FLAGS.print_every == 0:
                logging.info("Epoch %03d | Iteration %06d | Loss %.10f" % (e, i, loss))

        # Decay the scheduled sampling rate
        if FLAGS.sched_dec_type == 'lin':
            alternative = FLAGS.sched_k - FLAGS.sched_c * e
            FLAGS.sched_rate = max(FLAGS.sched_rate, alternative)
        elif FLAGS.sched_dec_type == 'exp':
            FLAGS.sched_rate = FLAGS.sched_k ^ e
        elif FLAGS.sched_dec_type == 'inverse_sig':
            divisor = FLAGS.sched_k + math.exp(e / FLAGS.sched_k)
            FLAGS.sched_rate = FLAGS.sched_k / divisor

        # Decrement the learning rate if the desired threshold has been surpassed
        if e > FLAGS.learning_rate_dec_thresh and i % FLAGS.learning_rate_dec_freq == 0:
            FLAGS.learning_rate /= FLAGS.learning_rate_dec_factor

        # Occasionally save model
        if e % FLAGS.save_every == 0:
            pass
            helpers.save_model(sess, saver, helpers.get_new_model_path())

    # Alert that training has been completed and print the run time
    elapsed = time.time() - start_time
    logging.info("Training complete. The session took %.2f seconds to complete." % elapsed)
    coord.request_stop()
    coord.join(threads)

    # Save the trained model join threads
    helpers.save_model(sess, saver, helpers.get_new_model_path(), trained=True)
    coord.request_stop()
    coord.join(threads)
