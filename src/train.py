"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import helpers
import logging
import tensorflow as tf
import time
from attention import Attention
from caption_extractor import CaptionExtractor
from caption_encoder import CaptionEncoder
from custom_vgg16 import Vgg16
from decoder import Decoder

BATCH_SIZE = 80
LEARNING_RATE = .0004
LEARNING_RATE_DEC = .8
LEARNING_RATE_DEC_FREQ = 3
LEARNING_RATE_DEC_THRESHOLD = 10
PRINT_EVERY = 100
TRAINING_ITERS = 100
TRAIN_HEIGHT = 512
TRAIN_WIDTH = 512

with tf.Session() as sess:
    image_shape = [1, TRAIN_HEIGHT, TRAIN_WIDTH, 3]

    # Initialize system instances
    extractor = CaptionExtractor()
    vgg = Vgg16()
    stv = CaptionEncoder()
    tatt = Attention()
    captioner = Decoder()

    # Initialize placeholders
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=image_shape)
    caption_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None])
    seq_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, ])
    learning_rate_placeholder = tf.placeholder(dtype=tf.float32, shape=[1])

    # Build encoder architectures
    vgg.build(image_placeholder, image_shape[1:])
    stv.build(caption_placeholder, seq_len_placeholder)

    #
    image_encoding = vgg.conv5_3
    guidance_caption = stv.output

    #
    tatt.build(image_encoding, guidance_caption)
    captioner.build()

    #
    output_caption = captioner.output

    loss = 0

    # Optimization ops
    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate_placeholder)
        update_step = 0

    example, label = helpers.next_example(height=TRAIN_HEIGHT, width=TRAIN_WIDTH)
    capacity = BATCH_SIZE * 2
    batch_condition, batch_label = tf.train.batch([example, label], BATCH_SIZE, num_threads=4, capacity=capacity)

    # Initialize session and threads then begin training
    logging.info("Begining training..")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()
    saver = tf.train.Saver()

    # Optimize
    for i in range(TRAINING_ITERS):
        # Initialize new feed dict for the training iteration and invoke the update op
        feed_dict = {learning_rate_placeholder: LEARNING_RATE}
        _, l = sess.run([update_step, loss], feed_dict=feed_dict)

        if i > LEARNING_RATE_DEC_THRESHOLD and i % LEARNING_RATE_DEC_FREQ == 0:
            LEARNING_RATE -= LEARNING_RATE_DEC

        if i % PRINT_EVERY == 0:
            logging.info("Epoch %06d | Loss %.06f" % (i, l))

    # Alert that training has been completed and print the run time
    elapsed = time.time() - start_time
    logging.info("Training complete. The session took %.2f seconds to complete." % elapsed)
    coord.request_stop()
    coord.join(threads)

    # Save the trained model and close the tensorflow session
    model_path = helpers.get_lib_path() + '/model_%s' % time.time()
    helpers.save_model(saver, model_path)
    sess.close()

exit(0)
