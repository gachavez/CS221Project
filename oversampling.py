import os
import glob
import torch
import numpy as np
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm

#let's read in our data, set our train, validation, test sets for good
#still using old data
directory = "/Users/emily/Desktop/Q4/CS221/Project/data/feature_extraction_output_v1/"
df = np.zeros([1,34])
count = 0
for file_name in glob.glob(directory+'*.csv'):
	count += 1
	curr_df = np.genfromtxt(file_name, delimiter = ",")
	curr_df = curr_df[:,:34]
	df = np.vstack((df,curr_df))
	print count
df = np.delete(df, (0), axis=0)

#X is features, y is ground truth
X = np.delete(df, [32,33], axis=1)
y = df[:,32]

#split training / validation / test sets (60-20-20 split between train, validation, test)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0, train_size=0.8)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 0, train_size = 0.75)

#oversample on training data
sm = SMOTE(random_state=0, ratio = 1.0)
x_res, y_res = sm.fit_sample(X_train, y_train)

#split training / validation / test sets (60-20-20 split between train, validation, test)
x_train_res, x_test_res, y_train_res, y_test_res = train_test_split(x_res,y_res, random_state = 0, train_size=0.8)
x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(x_train_res, y_train_res, random_state = 0, train_size = 0.75)

clf_rf = RandomForestClassifier(n_estimators=25, random_state=0)
clf_rf.fit(x_train_res, y_train_res)
clf_rf.score(x_val_res, y_val_res)
recall_score(y_val_res, clf_rf.predict(x_val_res))
confusion_matrix(clf_rf.predict(x_val_res), y_val_res)

#try SVM model
C = 1.0 #regularization parameter
svc_lin = svm.SVC(kernel = 'linear', C=C).fit(x_train_res, y_train_res)
svc_rbf = svm.SVC(kernel = 'rbf', gamma = 0.7, C=C).fit(x_train_res, y_train_res)
svc_poly = svm.SVC(kernel = 'poly', degree = 3, C=C).fit(x_train_res, y_train_res)
#clf_svm.fit(x_train_res, y_train_res) takes way too long to run :-(














