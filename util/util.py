from __future__ import division
import pickle as pk
from const import *

__author__ = 'hmourit'


def pickle_data(train_data, test_data, index = "0"):
    pk.dump(train_data, open(PICKLED_TRAIN_DATA % index, 'wb'))
    pk.dump(test_data, open(PICKLED_TEST_DATA % index, 'wb'))


def unpickle_data(index = "0"):
    train_data = pk.load(open(PICKLED_TRAIN_DATA % index, 'rb'))
    test_data = pk.load(open(PICKLED_TEST_DATA % index, 'rb'))
    return train_data, test_data


def get_parameters_hash(parameters):
    hash_ = str(hash(str(parameters)))
    with open(HASH_MEANING_FILE, 'a') as file_:
        file_.write('{},{}\n'.format(hash_, str(parameters)))
    return hash_