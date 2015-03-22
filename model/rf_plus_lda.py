from __future__ import division, print_function

import pickle as pk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from util.util import generate_submission
from util.const import *
from preprocessing.feature_engineering import engineer_data
from util.timer import Timer


p = {
    'features': ['date', 'time', 'season', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
                 'temp*humidity', 'temp*windspeed'],
    'target': TARGETS,
    'classifier': RandomForestRegressor,
    'classifier_args': {
        'n_estimators': 2000,
        'min_samples_split': 7,
        'oob_score': True
    },
    'lda': True
}

timer = Timer()

print('\n------------------')
print('Loading data... ', end='')
train_reg = pk.load(open("../data/lda_train_reg.pkl", 'rb'))
train_cas = pk.load(open("../data/lda_train_casual.pkl", 'rb'))
test_reg = pk.load(open("../data/lda_test_reg.pkl", 'rb'))
test_cas = pk.load(open("../data/lda_test_casual.pkl", 'rb'))

train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
X, y = engineer_data(train, p['features'], p['target'])

print('\n------------------')
print('Elapsed: {}'.format(timer.elapsed()))
rf = p['classifier'](**p['classifier_args'])
pred = np.zeros((test_cas.shape[0],))

print('\n------------------')
print('Training {} for "{}"... '.format(p['classifier'], 'registered'), end='')
train_reg.drop('registered', axis=1, inplace=True)
rf.fit(train_reg, y['registered'])
pred += rf.predict(test_reg)
print('Elapsed: {}'.format(timer.elapsed()))

print('\n------------------')
print('Training {} for "{}"... '.format(p['classifier'], 'casual'), end='')
train_cas.drop('casual', axis=1, inplace=True)
rf.fit(train_cas, y['casual'])
pred += rf.predict(test_cas)
print('Elapsed: {}'.format(timer.elapsed()))

print('\n------------------')
print('Generate submission')
pred = np.intp(pred.round())
generate_submission(test, pred, p)

print('\n------------------')
print('\nTotal time: {}\n\n'.format(timer.elapsed()))

