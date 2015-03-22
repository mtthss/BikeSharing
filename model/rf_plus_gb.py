from __future__ import division, print_function

from sklearn.ensemble import *

from util.util import *
from util.timer import Timer
from preprocessing.feature_engineering import engineer_data
from model import IntegratedRegressor, DayNightRegressor

# Adapted from https://github.com/dirtysalt/tomb/blob/master/kaggle/bike-sharing-demand/pub0.py

p = {
    'features': ['dummy_weekday', 'hour', 'year-2011', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                 'humidity', 'windspeed', 'night'],
    'template': IntegratedRegressor,
    'split_night': True,
    'target': TARGETS,
    'reg1': GradientBoostingRegressor,
    'reg1_args': {
        'n_estimators': 200,
        'max_depth': 4,
        'random_state': 0
    },
    'reg2': RandomForestRegressor,
    'reg2_args': {
        'n_estimators': 100,
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
        print('Training {}... '.format(p[reg_key]), end='')
        # for target in p['target']:
        #     reg.fit(X, np.log1p(y[target]))
        #     pred += np.expm1(reg.predict(X_test))
        reg = p[reg_key](**p[reg_key + '_args'])
        reg = IntegratedRegressor(reg)
        reg = DayNightRegressor(reg)
        reg.fit(X, y)
        pred += reg.predict(X_test)
        print('Elapsed: {}'.format(timer.elapsed()))

    # Average
    pred /= 2
    pred = np.intp(pred.round())

    generate_submission(test, pred, p)

    print('\nTotal time: {}\n\n'.format(timer.elapsed()))
