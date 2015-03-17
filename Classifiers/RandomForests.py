
###########
# Imports #
###########
import pandas as pd
import pickle
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestRegressor


########################
# Load Engineered Data #
########################
print "---------------"
print "Unpickling engineered data..."
pkl_train = open('../Data/train.pkl', 'rb')
pkl_test = open('../Data/test.pkl', 'rb')
pkl_cols = open('../Data/test.pkl', 'rb')
train = pickle.load(pkl_train)
test = pickle.load(pkl_test)
all_cols = pickle.load(pkl_test)

##################
# Random Forests #
##################
print "---------------"
print "Running Random Forests..."
cols = ['date','time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'tempTimesHumidity', 'tempTimesWindspeed', 'humidityTimesWindspeed']
rf = RandomForestRegressor(n_estimators=1500, min_samples_split=6, oob_score=True)

casual = rf.fit(train[cols], train.casual)
print casual.feature_importances_

predict_casual = rf.predict(test[cols])

registered = rf.fit(train[cols], train.registered)
print registered.feature_importances_

predict_registered = rf.predict(test[cols])

count = [int(round(i+j)) for i,j in zip(predict_casual, predict_registered)]


##############
# Submission #
##############
print "---------------"
print "Generating submission..."
df_submission = pd.DataFrame(count, test.datetime, columns = ['count'])
pd.DataFrame.to_csv(df_submission ,'../Data/rf_predict.csv')

