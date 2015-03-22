from __future__ import division, print_function

from random import sample, randint

from sklearn.ensemble import *
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

from util.util import *
from util.timer import Timer
from preprocessing.feature_engineering import engineer_data, ENGINEERED_FEATS
from model import IntegratedRegressor


# Adapted from https://github.com/dirtysalt/tomb/blob/master/kaggle/bike-sharing-demand/pub0.py

p = {
    'features': BASIC_FEAT + ENGINEERED_FEATS,
    'target': TARGETS,
    'reg': GradientBoostingRegressor,
    'reg_args': {
        'n_estimators': 100,
        'max_depth': 6,
        'random_state': 0
    },
    'uber_model': LinearRegression,
    'n_subsets': 10,
    'folds': 5,
    'seed': 42,
    'predict_log': True
}

if __name__ == '__main__':

    timer = Timer()

    print('Loading data... ', end='')
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    print('Elapsed: {}'.format(timer.elapsed()))

    Z_train = np.zeros((N_SAMPLES, p['n_subsets']))
    Z_test = np.zeros((N_TEST_SAMPLES, p['n_subsets']))
    reg = IntegratedRegressor(p['reg'], p['reg_args'], p['predict_log'])
    for i in range(p['n_subsets']):
        subset = sample(p['features'], randint(1, len(p['features'])))
        print('{} Training with {}'.format(timer.elapsed(), subset), end='\n')
        X, y = engineer_data(train, subset, p['target'])
        X_test = engineer_data(test, subset)
        y_hat = []
        for train_idx, test_idx in KFold(X.shape[0], n_folds=p['folds'], random_state=p['seed']):
            reg.fit(X.ix[train_idx, :], y.ix[train_idx, :])
            pred = reg.predict(X.ix[test_idx, :])
            y_hat = np.append(y_hat, pred)
        Z_train[:, i] = y_hat

        # Fit test
        reg.fit(X, y)
        y_hat = reg.predict(X_test)
        Z_test[:, i] = y_hat

    print()

    print('Weighing different subsets... ', end='')
    _, y = engineer_data(train, [], p['target'])
    # pred = np.zeros((N_TEST_SAMPLES,))
    # for target in p['target']:
    # reg = p['uber_model']()
    #     reg.fit(Z_train, np.log1p(y[target]))
    #     pred += np.expm1(reg.predict(Z_test))
    reg = IntegratedRegressor(p['uber_model'], {})
    reg.fit(Z_train, y)
    pred = reg.predict(Z_test)
    print('Elapsed: {}'.format(timer.elapsed()))

    # Average
    pred /= 2
    pred = np.intp(pred.round())

    generate_submission(test, pred, p)

    print('\nTotal time: {}\n\n'.format(timer.elapsed()))
