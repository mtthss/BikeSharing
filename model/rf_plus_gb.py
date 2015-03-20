from __future__ import division, print_function

from sklearn.ensemble import *
import numpy as np

from util.util import *
from util.timer import Timer
from Preprocessing.feature_engineering import engineer_data


# Adapted from https://github.com/dirtysalt/tomb/blob/master/kaggle/bike-sharing-demand/pub0.py

p = {
    'features': ['weekday', 'hour', 'year-2011', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                 'humidity', 'windspeed', 'day'],
    'target': TARGETS,
    'reg1': GradientBoostingRegressor,
    'reg1_args': {
        'n_estimators': 100,
        'max_depth': 6,
        'random_state': 0
    },
    'reg2': RandomForestRegressor,
    'reg2_args': {
        'n_estimators': 1000,
        'random_state': 0,
        'min_samples_split': 11,
        'oob_score': False,
        'n_jobs': -1
    }
}

if __name__ == '__main__':

    timer = Timer()

    print('Loading data... ', end='')
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    X, y = engineer_data(train, p['features'], p['target'])
    X_test = engineer_data(test, p['features'])
    print('Elapsed: {}'.format(timer.elapsed()))

    pred = np.zeros((X_test.shape[0],))
    for reg_key in ['reg1', 'reg2']:
        for target in p['target']:
            print('Training {} for "{}"... '.format(p[reg_key], target), end='')
            reg = p[reg_key](**p[reg_key + '_args'])
            reg.fit(X, np.log1p(y[target]))
            pred += np.expm1(reg.predict(X_test))
            print('Elapsed: {}'.format(timer.elapsed()))

    # Average
    pred /= 2
    pred = np.intp(pred.round())

    generate_submission(test, pred, p)

    print('\nTotal time: {}\n\n'.format(timer.elapsed()))