
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
train = pickle.load("../Data/train.pkl")
test = pickle.load("../Data/test.pkl")


##################
# Random Forests #
##################
cols = ['date','time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
rf = RandomForestRegressor(n_estimators=200)

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
df_submission = pd.DataFrame(count, test.datetime, columns = ['count'])
pd.DataFrame.to_csv(df_submission ,'randomforest_predict.csv')

