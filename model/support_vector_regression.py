from __future__ import division, print_function

import pandas as pd
from sklearn.svm import SVR
import numpy as np

from util.util import generate_submission
from util.const import *
from preprocessing.feature_engineering import engineer_data
from util.timer import Timer


p = {
    'features': ['date', 'time', 'season', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
                 'temp*humidity', 'temp*windspeed'],
    'target': TARGETS,
    'classifier': SVR,
    'classifier_args': {
         'C': 1.0,
         'tol': 0.01,
         'epsilon': 2.0,
         'cache_size': 2000,
         'kernel': 'sigmoid',
         'max_iter': 10
    }
}

timer = Timer()

print('Loading data... ', end='')
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
X, y = engineer_data(train, p['features'], p['target'])
X_test = engineer_data(test, p['features'])
print('Elapsed: {}'.format(timer.elapsed()))


rf = SVR(C=5.0, epsilon=0.2, kernel='poly')

#rf = p['classifier'](**p['classifier_args'])

pred = np.zeros((X_test.shape[0],))
for target in p['target']:
    print('Training {} for "{}"... '.format(p['classifier'], target), end='')
    rf.fit(X, y[target])
    pred += rf.predict(X_test)
    print('Elapsed: {}'.format(timer.elapsed()))

pred = np.intp(pred.round())

generate_submission(test, pred, p)

print('\nTotal time: {}\n\n'.format(timer.elapsed()))
