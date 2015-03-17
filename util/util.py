from __future__ import division
import pickle as pk
from const import *

__author__ = 'hmourit'


def pickle_data(train_data, test_data):
    pk.dump(train_data, open(PICKLED_TRAIN_DATA, 'wb'))
    pk.dump(test_data, open(PICKLED_TEST_DATA, 'wb'))


def unpickle_data():
    train_data = pk.load(open(PICKLED_TRAIN_DATA, 'rb'))
    test_data = pk.load(open(PICKLED_TEST_DATA, 'rb'))
    return train_data, test_data


def get_parameters_hash(parameters):
    hash_ = str(hash(str(parameters)))
    with open(HASH_MEANING_FILE, 'a') as file_:
        file_.write('{},{}\n'.format(hash_, str(parameters)))
    return hash_