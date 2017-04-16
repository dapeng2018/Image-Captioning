"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: This script is used to train a captioning model based on 'Text-guided Attention Model for Image Captioning'
"""

import helpers
import logging
import os
import tensorflow as tf
import time

import stv.configuration as stv_configuration
from attention import Attention
from caption_extractor import CaptionExtractor
from decoder import Decoder
from neighbor import Neighbor
from stv.encoder_manager import EncoderManager
from vgg.fcn16_vgg import FCN16VGG as Vgg16

FLAGS = tf.flags.FLAGS

# Optimization flags
tf.flags.DEFINE_integer('batch_size', 80, 'Mini-Batch size of images')
tf.flags.DEFINE_float('learning_rate', .0004, 'Optimizer learning rate')
tf.flags.DEFINE_float('learning_rate_dec_factor', .8, 'Factor in which the learning rate decreases')
tf.flags.DEFINE_integer('learning_rate_dec_freq', 3, 'How often (iterations) the learning rate decreases')
tf.flags.DEFINE_integer('learning_rate_dec_thresh', 10, 'Number of iterations before learning rate starts decreasing')

# Misc flags
tf.flags.DEFINE_integer('print_every', 100, 'How often (iterations) to log the current progress of training')
tf.flags.DEFINE_integer('save_every', 1000, 'How often (iterations) to save the current state of the model')

tf.flags.DEFINE_integer('conv_size', 1024, '')
tf.flags.DEFINE_float('dropout_rate', .5, '')
tf.flags.DEFINE_integer('embedding_size', 512, '')
tf.flags.DEFINE_integer('k', 10, 'Number of consensus captions to retrieve')
tf.flags.DEFINE_integer('kk', 16, '')
tf.flags.DEFINE_integer('n', 60, 'Number of nearest neighbors to retrieve')
tf.flags.DEFINE_integer('ngrams', 4, '')
tf.flags.DEFINE_integer('stv_size', 2400, '')
tf.flags.DEFINE_integer('training_iters', 100, 'Number of training iterations')
tf.flags.DEFINE_integer('train_height', 512, 'Height in which training images are to be scaled to')
tf.flags.DEFINE_integer('train_width', 512, 'Width in which training images are to be scaled to')
tf.flags.DEFINE_integer('train_height_sim', 224, 'Height in which images are to be scaled to for similarity comparison')
tf.flags.DEFINE_integer('train_width_sim', 224, 'Width in which images are to be scaled to for similarity comparison')
tf.flags.DEFINE_integer('vocab_size', 9568, 'Total size of vocabulary including <BOS> and <EOS>')

stv_lib = helpers.get_lib_path() + '/stv/'
tf.flags.DEFINE_string('stv_vocab_file', stv_lib + 'vocab.txt', 'Path to vocab file containing a list of words for STV')
tf.flags.DEFINE_string('stv_checkpoint_path', stv_lib + 'model.ckpt-501424', 'Path to STV model weights checkpoint')
tf.flags.DEFINE_string('stv_embeddings_file', stv_lib + 'embeddings.npy', 'Path to word embeddings for STV')

helpers.config_logging(env='training')


with tf.Session() as sess:
    image_shape = [1, FLAGS.train_height, FLAGS.train_width, 3]

    # Initialize system instances
    neighbor = Neighbor()
    vgg = Vgg16()
    stv = EncoderManager()

    # Skip thought model initialization
    stv_uni_config = stv_configuration.model_config()
    stv.load_model(stv_uni_config, FLAGS.stv_vocab_file, FLAGS.stv_embeddings_file, FLAGS.stv_checkpoint_path)

    # Initialize placeholders
    image_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.train_height, FLAGS.train_width, 3])
    image_name_ph = tf.placeholder(dtype=tf.string)
    image_fc_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, 7, 7, 4096])
    training_fc_encodings_ph = tf.placeholder(dtype=tf.float32, shape=[helpers.get_training_size(), 7, 7, 4096])
    training_filenames_ph = tf.placeholder(dtype=tf.string, shape=[helpers.get_training_size()])
    candidate_captions_ph = tf.placeholder(dtype=tf.string, shape=[FLAGS.n * FLAGS.k])
    caption_encoding_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.stv_size])
    label_ph = tf.placeholder(dtype=tf.int32, shape=[FLAGS.embedding_size])

    seq_len_ph = tf.placeholder(dtype=tf.int32, shape=[None, ])
    learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[1])

    # Build encoder architectures
    neighbor.build(image_fc_encoding_ph, training_fc_encodings_ph, training_filenames_ph)
    vgg.build(image_ph, image_shape[1:])
    conv_encoding = vgg.pool5
    fc_encoding = vgg.fc7
    extractor = CaptionExtractor(candidate_captions_ph)

    # Attention model and decoder
    tatt = Attention(conv_encoding, caption_encoding_ph)
    captioner = Decoder(tatt.context_vector)
    output_caption = captioner.output

    loss = 0

    # Optimization ops
    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate_ph)
        attention_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="attention")
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
        grads = optimizer.compute_gradients(loss, [attention_vars, decoder_vars])
        update_step = optimizer.apply_gradients(grads)

    # Training data ops
    example_image, example_filename = helpers.next_example(height=FLAGS.train_height, width=FLAGS.train_width)
    capacity = FLAGS.batch_size * 2
    batch_examples, batch_filenams = tf.train.batch([example_image, example_filename],
                                                    FLAGS.batch_size,
                                                    num_threads=4, capacity=capacity)
    all_examples, all_filenames = tf.train.batch([example_image, example_filename],
                                                 helpers.get_training_size(),
                                                 num_threads=4, capacity=capacity)

    all_examples_eval = all_examples.eval()
    all_filenames_eval = all_filenames.eval()

    # Initialize session and threads then begin training
    logging.info("Begining training..")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()
    saver = tf.train.Saver()

    # Optimize
    for i in range(FLAGS.training_iters):
        nearest_neighbors = neighbor.nearest.eval()
        candidate_captions = [extractor.captions[filename] for filename in nearest_neighbors]
        guidance_caption = extractor.guidance_caption.eval(feed_dict={})
        extractor.extend_to_len(guidance_caption, FLAGS.embedding_size)
        tokenized_caption = extractor.tokenize_sentence(guidance_caption)
        guidance_caption_encoding = stv.encode(tokenized_caption, batch_size=FLAGS.batch_size)

        # Initialize new feed dict for the training iteration and invoke the update op
        feed_dict = {learning_rate_ph: FLAGS.learning_rate, image_ph: batch_examples.eval()}
        _, l = sess.run([update_step, loss], feed_dict=feed_dict)
        sess.run(extractor.nearest_neighbors, feed_dict=feed_dict)

        # Decrement the learning rate if the desired threshold has been surpassed
        if i > FLAGS.learning_rate_dec_threh and i % FLAGS.learning_rate_dec_freq == 0:
            FLAGS.learning_rate /= FLAGS.learning_rate_dec_factor

        # Log loss
        if i % FLAGS.print_every == 0:
            logging.info("Iteration %06d | Loss %.06f" % (i, l))

    # Alert that training has been completed and print the run time
    elapsed = time.time() - start_time
    logging.info("Training complete. The session took %.2f seconds to complete." % elapsed)
    coord.request_stop()
    coord.join(threads)

    # Save the trained model and close the tensorflow sessions
    model_path = helpers.get_lib_path() + '/model_%s' % time.time()
    #helpers.save_model(saver, model_path)
    stv.close()
    sess.close()

exit(0)
