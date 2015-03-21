from __future__ import division, print_function

import pandas as pd

ENGINEERED_FEATS = ['year', 'year-2011', 'month', 'day', 'weekday', 'hour', 'dummy_season', 'temp * humidity',
                    'temp*windspeed', 'humidity*windspeed', 'temp/windspeed', 'dummy_weather', 'dummy_year',
                    'dummy_month', 'dummy_day', 'dummy_weekday', 'dummy_hour']


def _compute_features(data, features):
    df = pd.DataFrame()
    for feat in features:
        if feat in data:
            df[feat] = data[feat]
        elif feat == 'year':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.year)
        elif feat == 'dummy_year':
            df = pd.concat(
                [df, pd.get_dummies(pd.to_datetime(data['datetime']).apply(lambda x: x.year), prefix='year')], axis=1)
        elif feat == 'year-2011':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.year - 2011)
        elif feat == 'month':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.month)
        elif feat == 'dummy_month':
            dummies = pd.get_dummies(pd.to_datetime(data['datetime']).apply(lambda x: x.month), prefix='month')
            df = pd.concat([df, dummies], axis=1)
        elif feat == 'day':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.day)
        elif feat == 'dummy_day':
            dummies = pd.get_dummies(pd.to_datetime(data['datetime']).apply(lambda x: x.day), prefix='day')
            df = pd.concat([df, dummies], axis=1)
        elif feat == 'weekday':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.weekday())
        elif feat == 'dummy_weekday':
            dummies = pd.get_dummies(pd.to_datetime(data['datetime']).apply(lambda x: x.weekday()), prefix='weekday')
            df = pd.concat([df, dummies], axis=1)
        elif feat == 'hour':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.hour)
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
        elif feat == 'temp/windspeed':
            df[feat] = data['atemp'] / data['windspeed']
        elif feat == 'dummy_weather':
            df = pd.concat([df, pd.get_dummies(data['weather'], prefix='weather')], axis=1)
    return df


def engineer_data(data, features, target=None):
    if type(features) is str:
        features = [features]
    if type(target) is str:
        target = [target]
    X = _compute_features(data, features)
    if target is not None:
        y = _compute_features(data, target)
        return X, y
    else:
        return X
