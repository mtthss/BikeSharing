#################################
#################################
# TODO
#################################
#################################
# try lowering min_sample_split
# try increasing n_estimators
# try PCA
# try Theano MLP
#################################
#################################

###########
# Imports #
###########
from __future__ import division
import pandas as pd
import pickle
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestRegressor
from util.util import unpickle_data, get_parameters_hash
from util.const import *

########################
# Load Engineered Data #
########################
print "---------------"
print "Unpickling the engineered data..."
train, test = unpickle_data()

##################
# Random Forests #
##################

# parameters
p = {
    'cols': ['date', 'time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'],
    'classifier': RandomForestRegressor,
    'classifier_args': {
        'n_estimators': 1800,
        'min_samples_split': 7,
        'oob_score': True
    }
}

submission_hash = get_parameters_hash(p)

rf = p['classifier'](**p['classifier_args'])

casual = rf.fit(train[p['cols']], train.casual)
print casual.feature_importances_

predict_casual = rf.predict(test[p['cols']])

registered = rf.fit(train[p['cols']], train.registered)
print registered.feature_importances_

predict_registered = rf.predict(test[p['cols']])
count = [int(round(i + j)) for i, j in zip(predict_casual, predict_registered)]

##############
# Submission #
##############
print "---------------"
print "Generating submission..."
df_submission = pd.DataFrame(count, test.datetime, columns=['count'])
pd.DataFrame.to_csv(df_submission, PREDICTION_FILE % submission_hash)

