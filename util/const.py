from __future__ import division

__author__ = 'hmourit'

PICKLED_TRAIN_DATA = '../../Competition4Shared/Data/train_data_%s.pickle'
PICKLED_TEST_DATA = '../../Competition4Shared/Data/test_data_%s.pickle'

TRAIN_DATA = '../../Competition4Shared/Data/train.csv'
TEST_DATA = '../../Competition4Shared/Data/test.csv'

PREDICTION_FILE = '../../Competition4Shared/predictions/prediction_%s.csv'

HASH_MEANING_FILE = '../../Competition4Shared/hash_meaning.csv'

ALL_COLS = ['date', 'time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
            'tempTimesHumidity', 'tempTimesWindspeed', 'humidityTimesWindspeed', 'tempDivWindspeed']

ORIG_FEAT = [
    'datetime',
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'atemp',
    'humidity',
    'windspeed'
]

BASIC_FEAT = [
    'dummy_season',
    'holiday',
    'workingday',
    'dummy_weather',
    'temp',
    'atemp',
    'humidity',
    'windspeed'
]

TARGETS = [
    'casual',
    'registered'
]

ACTUAL_TARGET = 'count'

N_SAMPLES = 10886
N_TEST_SAMPLES = 6493