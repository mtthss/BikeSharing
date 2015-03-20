import pickle as pk

import numpy as np
import pandas as pd
from sklearn.lda import LDA

from util.util import unpickle_data



########################
# Load engineered data #
########################
print "\n---------------"
print "Unpickling the engineered data..."
cols = ['date', 'time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
train, test = unpickle_data()


##################
# Refactor input #
##################
print "\n---------------"
print "Refactoring..."
X_train_casual = np.array(train[cols])
X_test_casual = np.array(test[cols])
y_train_casual = np.array(train.casual)

X_train_reg = np.array(train[cols])
X_test_reg = np.array(test[cols])
y_train_reg = np.array(train.registered)


######################################
# Run linear discriminative analysis #
######################################
print "\n---------------"
print "Initialize LDA..."
clf = LDA()
dates_train = test.datetime
dates_test = train.datetime

print "\n---------------"
print "Run LDA wrt casual users..."
clf.fit(X_train_casual, y_train_casual)
LDA_train_casual = clf.transform(X_train_casual)
LDA_test_casual = clf.transform(X_test_casual)
df_LDA_train_casual = pd.DataFrame(LDA_train_casual, index=dates_train, columns=cols)
df_LDA_test_casual = pd.DataFrame(LDA_test_casual, index=dates_test, columns=cols)

print "\n---------------"
print "Run LDA wrt registered users..."
clf.fit(X_train_reg, y_train_reg)
LDA_train_reg = clf.transform(X_train_reg)
LDA_test_reg = clf.transform(X_test_reg)
df_LDA_train_reg = pd.DataFrame(LDA_train_reg, index=dates_train, columns=cols)
df_LDA_test_reg = pd.DataFrame(LDA_test_reg, index=dates_test, columns=cols)

print "\n---------------"
print "Run LDA wrt registered users..."
i = 0
for timestamp in df_LDA_train_reg['datetime']:
    i += 1
    df_LDA_train_reg.loc[i - 1, 'casual'] = y_train_reg[i - 1]
i = 0
for timestamp in df_LDA_train_casual['datetime']:
    i += 1
    df_LDA_test_reg.loc[i - 1, 'casual'] = y_train_casual[i - 1]


##########
# Pickle #
##########
print "---------------"
print "Pickling..."
pk.dump(df_LDA_test_casual, open("../Data/lda_test_casual.pkl", 'wb'))
pk.dump(df_LDA_test_reg, open("../Data/lda_test_reg.pkl", 'wb'))
pk.dump(df_LDA_train_casual, open("../Data/lda_train_casual.pkl", 'wb'))
pk.dump(df_LDA_train_reg, open("../Data/lda_train_casual.pkl", 'wb'))