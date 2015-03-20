from __future__ import division, print_function
import pickle as pk
from const import *
import pandas as pd
import numpy as np


def pickle_data(train_data, test_data, index="0"):
    pk.dump(train_data, open(PICKLED_TRAIN_DATA % index, 'wb'))
    pk.dump(test_data, open(PICKLED_TEST_DATA % index, 'wb'))


def unpickle_data(index="0"):
    train_data = pk.load(open(PICKLED_TRAIN_DATA % index, 'rb'))
    test_data = pk.load(open(PICKLED_TEST_DATA % index, 'rb'))
    return train_data, test_data


def _get_parameters_hash(parameters):
    hash_ = str(hash(str(parameters)))
    with open(HASH_MEANING_FILE, 'a') as file_:
        file_.write('{},{}\n'.format(hash_, str(parameters)))
    return hash_


def generate_submission(test, predictions, parameters, verbose=True):
    df_submission = pd.DataFrame(predictions, test.datetime, columns=['count'])
    hash_ = _get_parameters_hash(parameters)
    pd.DataFrame.to_csv(df_submission, PREDICTION_FILE % hash_)
    if verbose:
        print('Submission hash: {}'.format(hash_))


def rmsle(y_true, y_pred):
    n = y_true.shape[0]
    return np.sqrt(np.square(np.log1p(y_pred) - np.log1p(y_true)).sum() / n)
