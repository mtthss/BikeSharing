
###########
# Imports #
###########
import pandas as pd
from datetime import datetime
import numpy as np
np.set_printoptions(threshold=np.nan)


################
# Reading Data #
################
print "---------------"
print "Reading...!"
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


#######################
# Engineer Timestamps #
#######################
print "---------------"
print "Engineering timestamps!"
def transform_timestamps(df):

    i = 0
    for timestamp in df['datetime']:

        i += 1
        date_object = datetime.strptime(timestamp.split()[0], '%Y-%m-%d')
        time = timestamp.split()[1][:2]
        date = datetime.date(date_object).weekday()
        df.loc[i-1, 'date'] = date
        df.loc[i-1, 'time'] = time
    return df

train = transform_timestamps(df_train)
test = transform_timestamps(df_test)


##########
# Ending #
##########
print "---------------"
print "Done!"