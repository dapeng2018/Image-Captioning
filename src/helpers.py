"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: Contains various auxiliary functions.
"""

import logging
import numpy as np
import pickle
import os
import tensorflow as tf
import time
import skimage
import skimage.io
import skimage.transform
import sys
from vgg.fcn16_vgg import FCN16VGG as Vgg16

FLAGS = tf.flags.FLAGS


# Configure the python logger
def config_logging():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# Configure flags used for both training and inference
def config_model_flags():
    tf.flags.DEFINE_integer('conv_size', 512, 'Number of maps of the vgg image encoding')
    tf.flags.DEFINE_float('dropout_rate', .5, 'Probability of applying dropout for the final layer of the decoder')
    tf.flags.DEFINE_integer('embedding_size', 128, 'Max length of the embedding space')
    tf.flags.DEFINE_float('epsilon', 1e-8, 'Tiny value to for log parameters')
    tf.flags.DEFINE_integer('k', 10, 'Number of consensus captions to retrieve')
    tf.flags.DEFINE_integer('kk', 16 * 16, 'Filter map size (height * width) of the vgg image encoding')
    tf.flags.DEFINE_integer('max_caption_size', 30, 'Maximum number of words for caption if <eos> is not reached')
    tf.flags.DEFINE_integer('n', 60, 'Number of nearest neighbors to retrieve')
    tf.flags.DEFINE_integer('ngrams', 4, 'Number of grams (up-to) for candidate caption scoring')
    tf.flags.DEFINE_float('sched_rate', .75, 'Selection probability for scheduled sampling')
    tf.flags.DEFINE_integer('stv_size', 2400, 'Output size of the skip-thought vector encoder')
    tf.flags.DEFINE_integer('train_height', 512, 'Height in which training images are to be scaled to')
    tf.flags.DEFINE_integer('train_width', 512, 'Width in which training images are to be scaled to')
    tf.flags.DEFINE_integer('train_height_sim', 224, 'Height that images are to be scaled to for similarity comparison')
    tf.flags.DEFINE_integer('train_width_sim', 224, 'Width that images are to be scaled to for similarity comparison')
    tf.flags.DEFINE_integer('vocab_size', 9568, 'Total size of vocabulary including <bos> and <eos>')

    # Skip thought vector model flags
    stv_lib = get_lib_path() + '/stv/'
    tf.flags.DEFINE_string('stv_vocab_file', stv_lib + 'vocab.txt', 'Path to vocab file containing STV word list')
    tf.flags.DEFINE_string('stv_checkpoint_path', stv_lib + 'model.ckpt-501424', 'Path to STV model weights checkpoint')
    tf.flags.DEFINE_string('stv_embeddings_file', stv_lib + 'embeddings.npy', 'Path to word embeddings for STV')


# Retirve the absolute dir of the MSCOCO annotations dir
def get_annotations_path():
    return get_lib_path() + '/annotations/'


# Retirve the absolute path of the MSCOCO training caption set
def get_captions_path(train=True):
    if train:
        return get_annotations_path() + '/captions_train2014.json'
    else:
        return get_annotations_path() + '/captions_val2014.json'


# Measures cosine similarity for two given image encodings from the last FC layer of the VGG16
def get_cosine_similarity(a, b):
    shape = [-1, FLAGS.kk * 4096]
    a = tf.reshape(a, shape=shape)
    b = tf.reshape(b, shape=shape)
    ab = tf.matmul(a, b, transpose_b=True)

    a_norm = tf.nn.l2_normalize(a, dim=1)
    b_norm = tf.nn.l2_normalize(b, dim=1)
    ab_norm = tf.matmul(a_norm, b_norm, transpose_b=True)

    similarity = ab / ab_norm
    return similarity


# Retrieve the current absolute path of the script being executed
def get_current_path():
    return os.path.dirname(os.path.realpath(__file__)) + '/../'


# Retrieve a dictionary from a specified pickle file if it exists
def get_data(name):
    if pickle_exists(name):
        return load_obj(name)
    else:
        return {}


# Generate a configuration for a TensorFlow session
def get_session_config():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
    return tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)


# Retrieve the absolute path of the lib dir
def get_lib_path():
    return get_current_path() + '/lib/'


# Retrieve the absolute path of the log dir
def get_logs_path():
    return get_current_path() + '/log/'


# Generate a new model path based on the time
def get_new_model_path():
    return get_lib_path() + '/models/model_%s' % time.time()


# Retrieve the absolute path of the training dir
def get_training_path():
    return get_lib_path() + '/train2014/'


# Retrieve the number of training images available in training dir
def get_training_size():
    path = get_training_path()
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


# Retrieve a path for a training image
def get_training_next_path():
    for root, dirs, files in os.walk(get_training_path()):
        for name in files:
            if ".jpg" in name:
                return os.path.join(root, name)
            else:
                return get_training_next_path()


# Convert some index into a one-hot representation (list)
def index_to_1hot(index):
    return [1 if i == index else 0 for i in range(FLAGS.vocab_size)]


# Returns a numpy array of an image specified by its path
def load_img(path):
    # Load image [height, width, depth]
    img = skimage.io.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    shape = list(img.shape)

    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (shape[0], shape[1]))
    return resized_img, shape


# Return a resized numpy array of an image specified by its path
def load_image_to(path, height=None, width=None):
    # Load image
    img = skimage.io.imread(path) / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


# Retrieve the next training example and its filename with the dimensions specified
def next_example(height, width):
    # Ops for getting training images, from retrieving the filenames to reading the data
    regex = get_training_path() + '*.jpg'
    filenames = tf.train.match_filenames_once(regex)
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    filename, file = reader.read(filename_queue)

    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.image.resize_images(img, [height, width])

    return img, filename


# Log a python object from a pickle file in the lib dir by simply specifying its name (minus .pkl)
def load_obj(name):
    with open(get_lib_path() + name + '.pkl', 'rb') as f:
        x = pickle.load(f)
        return x


# Return whether or not a specified pickle file exists in the lib dir
def pickle_exists(name):
    return os.path.exists(get_lib_path() + name + '.pkl')


# Saves the model weights
def save_model(session, saver, path, trained=False):
    logging.info("Proceeding to save weights at '%s'" % path)

    # Prepend stamp to indicate this is a trained model
    if trained:
        path = "trained_" + path

    saver.save(session, path)
    logging.info("Weights have been saved.")


# Save a python object as a pickle file in the lib dir
def save_obj(obj, name):
    with open(get_lib_path() + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

