from __future__ import division, print_function

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor

from util.util import generate_submission
from util.const import *
from preprocessing.feature_engineering import engineer_data
from util.timer import Timer
from model import IntegratedRegressor

import argparse
import json

import os


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


def _process_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if key == 'base_estimator' and value == 'BayesianRidge':
            value = globals()[value](fit_intercept=True)
        elif key == 'base_estimator' and value == 'RandomForest':
            value = globals()[value](fit_intercept=True)
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
    train = pd.read_csv(TRAIN_DATA[3:])
    test = pd.read_csv(TEST_DATA[3:])
    features = config['features'].split(', ')
    X, y = engineer_data(train, features, TARGETS, normalize=True)
    X_test = engineer_data(test, features, normalize=True)

    print('Elapsed: {}'.format(timer.elapsed()))

    #################################
    ### Create a classifier model ###
    #################################
    arguments = _process_dict(config['classifier_args'])
    predictor = locals()[config['classifier']](**arguments)
    if 'predict_log' in config and config['predict_log'] == "False":
        predict_log = False
    else:
        predict_log = True
    predictor = IntegratedRegressor(predictor, predict_log=predict_log)
    

    #################################
    ### Train a classifier model  ###
    #################################
    print('{} Training {}... '.format(timer.elapsed(), config['classifier']))
    predictor.fit(X, y)
    pred = predictor.predict(X_test)
    # round and convert to int
    pred = np.intp(pred.round())

    #############################
    ### Write the predictions ###
    #############################
    os.chdir('./model')
    generate_submission(test, pred, config)

    print('\nTotal time: {}\n\n'.format(timer.elapsed()))
