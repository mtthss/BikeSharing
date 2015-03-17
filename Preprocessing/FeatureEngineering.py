###########
# Imports #
###########
from __future__ import division
from datetime import datetime

import pandas as pd
import numpy as np


np.set_printoptions(threshold=np.nan)
from util.const import *
from util.util import pickle_data


################
# Reading Data #
################
print "---------------"
print "Reading..."
df_train = pd.read_csv(TRAIN_DATA)
df_test = pd.read_csv(TEST_DATA)


#######################
# Engineer Timestamps #
#######################
print "\n---------------"
print "Engineering timestamps..."


def transform_timestamps(df):
    i = 0
    for timestamp in df['datetime']:
        i += 1
        date_object = datetime.strptime(timestamp.split()[0], '%Y-%m-%d')
        time = timestamp.split()[1][:2]
        date = datetime.date(date_object).weekday()
        df.loc[i - 1, 'date'] = date
        df.loc[i - 1, 'time'] = time
    return df


train = transform_timestamps(df_train)
test = transform_timestamps(df_test)


##############################
# Combining weather features #
##############################
print "\n---------------"
print "Combining weather features..."


def my_poly(df):
    i = 0
    for temp in df['atemp']:
        i += 1
        hu = df.loc[i - 1, 'humidity']
        ws = df.loc[i - 1, 'windspeed']
        tempTimesHumidity = temp * hu
        tempTimesWindspeed = temp * ws
        humidityTimesWindSpeed = hu * ws
        tempDivWindspeed = float(temp) / ws
        df.loc[i - 1, 'tempTimesHumidity'] = tempTimesHumidity
        df.loc[i - 1, 'tempTimesWindspeed'] = tempTimesWindspeed
        df.loc[i - 1, 'humidityTimesWindspeed'] = humidityTimesWindSpeed
        df.loc[i - 1, 'tempDivWindspeed'] = tempDivWindspeed

    return df


train = my_poly(df_train)
test = my_poly(df_test)


#################
# Insert memory #
#################

# previous day temp and humidity


##########
# Pickle #
##########
print '\n------------------'
print 'Pickling...'

pickle_data(train, test)

##########
# Ending #
##########
print "\n---------------"
print "Done!"