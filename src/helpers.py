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


def config_logging(env='training'):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def config_model_flags():
    tf.flags.DEFINE_integer('conv_size', 512, 'Number of maps of the vgg image encoding')
    tf.flags.DEFINE_float('dropout_rate', .5, 'Probability of applying dropout for the final layer of the decoder')
    tf.flags.DEFINE_integer('embedding_size', 512, 'Max length of the embedding space')
    tf.flags.DEFINE_integer('k', 10, 'Number of consensus captions to retrieve')
    tf.flags.DEFINE_integer('kk', 16 * 16, 'Filter map size (height * width) of the vgg image encoding')
    tf.flags.DEFINE_integer('max_caption_size', 30, 'Maximum number of words for caption if <eos> is not reached')
    tf.flags.DEFINE_integer('n', 60, 'Number of nearest neighbors to retrieve')
    tf.flags.DEFINE_integer('ngrams', 4, 'Number of grams (up-to) for candidate caption scoring')
    tf.flags.DEFINE_float('sched_rate', .75, 'Selection probability for scheduled sampling')
    tf.flags.DEFINE_integer('state_size', 512, 'State size of the LSTM')
    tf.flags.DEFINE_integer('stv_size', 2400, '')
    tf.flags.DEFINE_integer('training_iters', 100, 'Number of training iterations')
    tf.flags.DEFINE_integer('train_height', 512, 'Height in which training images are to be scaled to')
    tf.flags.DEFINE_integer('train_width', 512, 'Width in which training images are to be scaled to')
    tf.flags.DEFINE_integer('train_height_sim', 224, 'Height that images are to be scaled to for similarity comparison')
    tf.flags.DEFINE_integer('train_width_sim', 224, 'Width that images are to be scaled to for similarity comparison')
    tf.flags.DEFINE_integer('vocab_size', 9568, 'Total size of vocabulary including <BOS> and <EOS>')

    # Skip thought vector model flags
    stv_lib = get_lib_path() + '/stv/'
    tf.flags.DEFINE_string('stv_vocab_file', stv_lib + 'vocab.txt', 'Path to vocab file containing STV word list')
    tf.flags.DEFINE_string('stv_checkpoint_path', stv_lib + 'model.ckpt-501424', 'Path to STV model weights checkpoint')
    tf.flags.DEFINE_string('stv_embeddings_file', stv_lib + 'embeddings.npy', 'Path to word embeddings for STV')


def get_annotations_path():
    return get_lib_path() + '/annotations/'


def get_captions_path(train=True):
    if train:
        return get_annotations_path() + '/captions_train2014.json'
    else:
        return get_annotations_path() + '/captions_val2014.json'


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


def get_current_path():
    return os.path.dirname(os.path.realpath(__file__)) + '/../'


def get_data(name):
    if pickle_exists(name):
        return load_obj(name)
    else:
        return {}


def get_fc7_encodings():
    encoding_filename = 'fc7_encodings.pkl'
    encoding_path = get_lib_path() + encoding_filename
    encodings = np.load(encoding_path)

    return encodings


def get_fc7_filenames():
    image_filename = 'fc7_filenames.pkl'
    image_path = get_lib_path() + image_filename
    filenames = np.load(image_path)
    return filenames


def get_session_config():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    return tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)


def get_lib_path():
    return get_current_path() + '/lib/'


def get_logs_path():
    return get_current_path() + '/log/'


def get_new_model_path():
    return get_lib_path() + '/model_%s' % time.time()


def get_training_path():
    return get_lib_path() + '/train2014/'


def get_training_size():
    path = get_training_path()
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


def get_training_next_path():
    for root, dirs, files in os.walk(get_training_path()):
        for name in files:
            if ".jpg" in name:
                return os.path.join(root, name)
            else:
                return get_training_next_path()


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


def load_obj(name):
    with open(get_lib_path() + name + '.pkl', 'rb') as f:
        x = pickle.load(f)
        return x


def make_training_fc7_file():
    save_obj([], 'fc7_filenames')
    save_obj([], 'fc7_encodings')

    # Append a dictionary object {filename: encoding} to the list contained in the pickle file
    def append_fc7_encoding(f, e):
        filenames = get_fc7_filenames()
        filenames.append(f)
        save_obj(filenames, 'fc7_filenames')

        encodings = get_fc7_encodings()
        encodings.append(e)
        save_obj(encodings, 'fc7_encodings')

    # Create a tf session to evaluate encodings for training images resized to the given shape
    shape = [1, 224, 224, 3]
    with tf.Session() as sess:
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=shape)

        vgg = Vgg16()
        vgg.build(image_placeholder, shape[1:])
        layer = vgg.fc7

        sess.run(tf.global_variables_initializer())

        # Iterate through the training set, evaluate the encodings, and append to the pickle file
        for i in range(get_training_size()):
            if i % 100 == 0:
                print(i)
            filename = get_training_next_path()
            img = load_image_to(filename, shape[1], shape[2])
            img = img.reshape(shape).astype(np.float32)
            encoding = sess.run(layer, feed_dict={image_placeholder: img})
            append_fc7_encoding(filename, encoding)


def pickle_exists(name):
    return os.path.exists(get_lib_path() + name + '.pkl')


# Saves the model weights
def save_model(session, saver, path):
    logging.info("Proceeding to save weights at '%s'" % path)
    saver.save(session, path)
    logging.info("Weights have been saved.")


def save_obj(obj, name):
    with open(get_lib_path() + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

