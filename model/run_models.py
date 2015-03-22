from __future__ import division, print_function

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from util.util import generate_submission
from util.const import *
from preprocessing.feature_engineering import engineer_data
from util.timer import Timer

import argparse
import json


def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv


def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv


if __name__ == '__main__':

    #################################
    ### Parse arguments from JSON ###
    #################################
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename")
    config = parser.parse_args()
    config = json.load(open(config.filename), object_hook=_decode_dict)

    timer = Timer()
    print('Loading data... ', end='')

    #####################
    ### Load the data ###
    #####################
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    features = config['features'].split(', ')
    X, y = engineer_data(train, features, TARGETS)
    X_test = engineer_data(test, features)

    print('Elapsed: {}'.format(timer.elapsed()))

    #################################
    ### Create a classifier model ###
    #################################
    predictor = locals()[config['classifier']](**config['classifier_args'])

    #################################
    ### Train a classifier model  ###
    #################################
    pred = np.zeros((X_test.shape[0],))
    for target in TARGETS:
        print('Training {} for "{}"... '.format(config['classifier'], target), end='')
        predictor.fit(X, y[target])
        pred += predictor.predict(X_test)
        print('Elapsed: {}'.format(timer.elapsed()))
    # round and convert to int
    pred = np.intp(pred.round())

    #############################
    ### Write the predictions ###
    #############################
    generate_submission(test, pred, config)

    print('\nTotal time: {}\n\n'.format(timer.elapsed()))
