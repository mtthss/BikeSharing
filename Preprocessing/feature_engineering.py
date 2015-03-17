from __future__ import division, print_function

import pandas as pd


def engineer_data(data, features):
    df = pd.DataFrame()
    for feat in features:
        if feat in data:
            df[feat] = data[feat]
        elif feat == 'datetime':
            df[feat] = pd.to_datetime(data['datetime'])
        elif feat == 'weekday':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.weekday())
        elif feat == 'hour':
            df[feat] = pd.to_datetime(data['datetime']).apply(lambda x: x.hour)
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

    return df

