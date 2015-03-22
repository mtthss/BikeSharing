from __future__ import division, print_function

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from util.util import generate_submission
from util.const import *
from preprocessing.feature_engineering import engineer_data
from util.timer import Timer


p = {
    'features': ['date', 'time', 'season', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
                 'temp*humidity', 'temp*windspeed'],
    'target': TARGETS,
    'classifier': KNeighborsRegressor,
    'classifier_args': {
        'n_neighbors': 7,
        'weights':'distance',
        'algorithm': 'auto',
    }
}

timer = Timer()

print('Loading data... ', end='')
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
X, y = engineer_data(train, p['features'], p['target'])
X_test = engineer_data(test, p['features'])
print('Elapsed: {}'.format(timer.elapsed()))

"""
from sklearn.kernel_ridge import KernelRidge
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> clf = KernelRidge(alpha=1.0)
>>> clf.fit(X, y)
KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='linear',
            kernel_params=None)
"""




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
