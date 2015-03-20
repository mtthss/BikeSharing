from __future__ import division, print_function

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from util.util import generate_submission
from util.const import *
from Preprocessing.feature_engineering import engineer_data
from util.timer import Timer


p = {
    'features': ['date', 'time', 'season', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
                 'temp*humidity', 'temp*windspeed'],
    'target': TARGETS,
    'classifier': RandomForestRegressor,
    'classifier_args': {
        'n_estimators': 1800,
        'min_samples_split': 7,
        'oob_score': True
    }
}

timer = Timer()

print('Loading data... ', end='')
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
X, y = engineer_data(train, p['features'], p['target'])
X_test = engineer_data(test, p['features'])
print('Elapsed: {}'.format(timer.elapsed()))

rf = p['classifier'](**p['classifier_args'])

pred = np.zeros((X_test.shape[0],))
for target in p['target']:
    print('Training {} for "{}"... '.format(p['classifier'], target), end='')
    rf.fit(X, y[target])
    pred += rf.predict(X_test)
    print('Elapsed: {}'.format(timer.elapsed()))

pred = np.intp(pred.round())

generate_submission(test, pred, p)

print('\nTotal time: {}\n\n'.format(timer.elapsed()))

