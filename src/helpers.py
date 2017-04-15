"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import json
import logging
import numpy as np
import os
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
from vgg.fcn16_vgg import FCN16VGG as Vgg16


def get_annotations_path():
    return get_lib_path() + '/annotations/'


def get_captions_path(train=True):
    if train:
        return get_annotations_path() + '/captions_train2014.json'
    else:
        return get_annotations_path() + '/captions_val2014.json'


def get_cosine_similarity(a, b):
    a_norm = get_l2_norm(a)
    b_norm = get_l2_norm(b)
    ab = tf.matmul(b, a, transpose_b=True)
    similarity = ab / (a_norm * b_norm)
    return similarity


def get_current_path():
    return os.path.dirname(os.path.realpath(__file__)) + '/../'


def get_data(name):
    if json_exists(name):
        return load_obj(name)
    else:
        return {}


def get_lib_path():
    return get_current_path() + '/lib/'


def get_l2_norm(t):
    t = tf.squeeze(t)
    norm = tf.sqrt(tf.reduce_sum(tf.square(t)))
    return norm


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
    regex = get_training_path() + '/*.jpg'
    filenames = tf.train.match_filenames_once(regex)
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    filename, file = reader.read(filename_queue)

    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.image.resize_images(img, [height, width])

    return img, filename


def json_exists(name):
    return os.path.exists(get_lib_path() + name + '.json')


def save_model(session, saver, path):
    logging.info("Proceeding to save weights at '%s'" % path)
    saver.save(session, path)
    logging.info("Weights have been saved.")


def save_obj(obj, name):
    with open(get_lib_path() + name + '.json', 'w') as f:
        json.dump(obj, f)


def load_obj(name):
    with open(get_lib_path() + name + '.json', 'r') as f:
        x = json.load(f)
        return x


def make_training_fc7_file():
    # Append a dictionary object {filename: encoding} to the list contained in the json file
    def append_fc7_encoding(x, p):
        with open(p, 'w') as out:
            json.dump(x, out)

    # Initialize the empty json file
    filename = 'fc7_encoding.json'
    path = get_lib_path() + filename
    with open(path, mode='w', encoding='utf-8') as f:
        json.dump([], f)

    # Create a tf session to evaluate encodings for training images resized to the given shape
    shape = [1, 224, 224, 3]
    with tf.Session() as sess:
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=shape)

        vgg = Vgg16()
        vgg.build(image_placeholder, shape[1:])
        layer = vgg.fc7

        sess.run(tf.global_variables_initializer())

        # Iterate through the training set, evaluate the encodings, and append to the json file
        for _ in range(get_training_size()):
            filename = get_training_next_path()
            img = load_image_to(filename, shape[1], shape[2])
            img = img.reshape(shape).astype(np.float32)
            encoding = sess.run(layer, feed_dict={image_placeholder: img})
            append_fc7_encoding({filename: encoding}, path)
