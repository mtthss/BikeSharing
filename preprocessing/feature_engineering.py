from __future__ import division, print_function

from datetime import date

import pandas as pd


ENGINEERED_FEATS = ['year', 'year-2011', 'month', 'weekday', 'hour', 'dummy_season', 'temp * humidity',
                    'temp*windspeed', 'humidity*windspeed', 'dummy_weather', 'dummy_year',
                    'dummy_month', 'dummy_weekday', 'dummy_hour', 'hour/2', 'night']


def _compute_features(data, features, **kwargs):
    df = pd.DataFrame()
    for feat in features:
        if feat in data:
            df[feat] = data[feat]
        elif feat == 'year':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.year)
        elif feat == 'dummy_year':
            dummies = pd.get_dummies(pd.to_datetime(data['datetime']).apply(lambda x: x.year), prefix='year')
            df = pd.concat([df, dummies], axis=1)
        elif feat == 'year-2011':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.year - 2011)
        elif feat == 'month':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.month)
        elif feat == 'dummy_month':
            dummies = pd.get_dummies(pd.to_datetime(data['datetime']).apply(lambda x: x.month), prefix='month')
            df = pd.concat([df, dummies], axis=1)
        elif feat == 'weekday':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.weekday())
        elif feat == 'dummy_weekday':
            dummies = pd.get_dummies(pd.to_datetime(data['datetime']).apply(lambda x: x.weekday()), prefix='weekday')
            df = pd.concat([df, dummies], axis=1)
        elif feat == 'hour':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.hour)
        elif feat == 'hour/2':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: int(x.hour / 2))
        elif feat == 'dummy_hour':
            dummies = pd.get_dummies(pd.to_datetime(data['datetime']).apply(lambda x: x.hour), prefix='hour')
            df = pd.concat([df, dummies], axis=1)
        elif feat == 'dummy_season':
            df['spring'] = (data['season'] == 1).astype(int)
            df['summer'] = (data['season'] == 2).astype(int)
            df['fall'] = (data['season'] == 3).astype(int)
            df['winter'] = (data['season'] == 4).astype(int)
        elif feat == 'temp*humidity':
            df[feat] = data['atemp'] * data['humidity']
        elif feat == 'temp*windspeed':
            df[feat] = data['atemp'] * data['windspeed']
        elif feat == 'humidity*windspeed':
            df[feat] = data['atemp'] * data['windspeed']
        elif feat == 'dummy_weather':
            df = pd.concat([df, pd.get_dummies(data['weather'], prefix='weather')], axis=1)
        elif feat == 'night':
            if 'night_range' not in kwargs:
                night_range = range(1, 7)
            else:
                night_range = kwargs['night_range']
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.hour in night_range).astype(int)
        elif feat == 'days_passed':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: (x.date() - date(2011, 1, 1)).days).astype(int)
            # TODO day since the beginning
    return df


def engineer_data(data, features, target=None, normalize=False, **kwargs):
    if type(features) is str:
        features = [features]
    if type(target) is str:
        target = [target]
    X = _compute_features(data, features, **kwargs)
    if normalize:
        X = (X - X.mean()) / (X.max() - X.min())
    if target is not None:
        y = _compute_features(data, target)
        return X, y
    else:
        return X
