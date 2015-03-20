from __future__ import division, print_function

from sklearn.linear_model import MultiTaskLassoCV

from util.util import *
from util.timer import Timer
from Preprocessing.feature_engineering import engineer_data

p = {
    # 'features': BASIC_FEAT + ['year', 'month', 'day', 'hour'],
    'features': ['weekday', 'hour', 'year-2011', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                 'humidity', 'windspeed', 'day'],
    'target': TARGETS,
    'reg': MultiTaskLassoCV,
    'reg_args': {
        'cv': 5,
        'n_jobs': -1
    }
}

timer = Timer()

print('Loading data... ', end='')
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
X, y = engineer_data(train, p['features'], TARGETS)
y['casual'] = np.log1p(y['casual'])
y['registered'] = np.log1p(y['registered'])
X_test = engineer_data(test, p['features'])
print('Elapsed: {}'.format(timer.elapsed()))

print('Training regressor... ', end='')
reg = p['reg'](**p['reg_args'])
reg.fit(X, y)
print('Elapsed: {}'.format(timer.elapsed()))

print('Generate results... ', end='')
result = reg.predict(X_test)
result = np.intp(np.expm1(result).sum(axis=1).round())
result[result < 0] = 0
print('Elapsed: {}'.format(timer.elapsed()))

generate_submission(test, result, p)


