##read in files
import os
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split


#import files to create dataframe

#read in all csv files created by "feature_extraction.py"
#set to your directory to wherever the "feature_extraction_output/" folder is
directory = "/Users/emily/Desktop/Q4/CS221/Project/CS221Project-testing-PyEDF-library/feature_extraction_output/"
df = np.zeros([1,34])
count = 0
for file_name in glob.glob(directory+'*.csv'):
	#count += 1
	curr_df = np.genfromtxt(file_name, delimiter = ",")
	curr_df = curr_df[:,:34]
	df = np.vstack((df,curr_df))
	#print count
df = np.delete(df, (0), axis=0) #because i initialized with a row of zeroes, sorry hacky
df.shape #(1744046, 34)

#get sum of row for baseline feature, add to dataframe
baseline_feat = df.sum(1)[...,None]
df2 = np.hstack((df,baseline_feat))

#Separate into test and training data, this should be preliminary, we should
#talk about how to handle the rarity of positive labels
X = np.delete(df2, 32, axis=1)
y = df2[:,32]

#we do get a good sample of positive labels, despite rarity of event
#we'll want to split into test, train, validation...
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0, train_size=0.8)

#models:
feat1_train, feat1_test = X_train[:,33], X_test[:,33]
feat1_train = feat1_train.reshape(-1,1)
feat1_test = feat1_test.reshape(-1,1)
oracle_train, oracle_test = X_train[:,32], X_test[:,32]
oracle_train = oracle_train.reshape(-1,1)
oracle_test = oracle_test.reshape(-1,1)

baseline_logistic = LogisticRegression(random_state = 0)
baseline_logistic.fit(feat1_train, y_train)


baseline_predicts = baseline_logistic.predict(feat1_test)
baseline_cm = confusion_matrix(baseline_predicts, y_test)
tn, fp, fn, tp = baseline_cm.ravel() 
misclass_err = float(1069 +51) / (51 + 347689 + 1069) #0.0032109263235753664
baseline_pos = float(tp) / (tp + fp)


oracle_logistic = LogisticRegression(random_state = 0)
oracle_logistic.fit(oracle_train, y_train)

oracle_predicts = oracle_logistic.predict(oracle_test)
oracle_cm = confusion_matrix(oracle_predicts, y_test)
tn, fp, fn, tp = oracle_cm.ravel()
misclass_err = float(fp + fn) / (tn + tp + fn + fp) #0.0030647144999125598
oracle_sensitivity = float(tp) / (tp + fn)

oracle_logistic2 = LogisticRegression(random_state = 0)
oracle_logistic2.fit(X_train, y_train)
oracle_predicts2 = oracle_logistic2.predict(X_test)
oracle_cm2 = confusion_matrix(oracle_predicts2, y_test)












