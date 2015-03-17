from __future__ import division, print_function
from sklearn.ensemble import *
from util.util import unpickle_data, get_parameters_hash, generate_submission
from util.timer import Timer
import numpy as np

__author__ = 'hmourit'

p = {
    'cols': ['date', 'time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'],
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
    submission_hash = get_parameters_hash(p)
    train, test = unpickle_data()

    timer = Timer()

    # Reg1
    print('Training Gradient Boosting Regressor... ', end='')
    reg1 = p['reg1'](**p['reg1_args'])
    reg1.fit(train[p['cols']], np.log1p(train.casual))
    pred = np.expm1(reg1.predict(test[p['cols']]))
    reg1.fit(train[p['cols']], np.log1p(train.registered))
    pred += np.expm1(reg1.predict(test[p['cols']]))
    print('finished in {}'.format(timer.elapsed()))

    # Reg2
    print('Training Random Forest Regressor... ', end='')
    reg2 = p['reg2'](**p['reg2_args'])
    reg2.fit(train[p['cols']], np.log1p(train.casual))
    pred += np.expm1(reg1.predict(test[p['cols']]))
    reg2.fit(train[p['cols']], np.log1p(train.registered))
    pred += np.expm1(reg1.predict(test[p['cols']]))
    print('finished in {}'.format(timer.elapsed()))

    # Average
    pred /= 2
    pred = np.intp(pred.round())

    generate_submission(test, pred, submission_hash)

    print('\nTotal time: {}\n\n'.format(timer.elapsed()))
