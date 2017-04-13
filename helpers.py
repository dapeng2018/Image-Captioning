"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description:
"""

import os


def get_annotations_path():
    return get_lib_path() + '/annotations/'


def get_captions_path(train=True):
    if train:
        return get_annotations_path() + '/captions_train2014.json'
    else:
        return get_annotations_path() + '/captions_val2014.json'


def get_current_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_lib_path():
    return get_current_path() + '/lib/'
