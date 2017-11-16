import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

train = np.genfromtxt("/Users/emily/Desktop/Q4/CS221/Project/data/separated/epilepsy_train.csv", delimiter = ",")
val = np.genfromtxt("/Users/emily/Desktop/Q4/CS221/Project/data/separated/epilepsy_validation.csv", delimiter = ",")
test = np.genfromtxt("/Users/emily/Desktop/Q4/CS221/Project/data/separated/epilepsy_test.csv", delimiter = ",")

#last column is ground truth
#second to last is oracle
x_train = np.delete(train, [32,33], axis=1)
y_train =  train[:,33]
train_oracle = train[:,32]
x_val = np.delete(val, [32,33], axis=1)
y_val = val[:,33]
val_oracle = val[:,32]
x_test = np.delete(test, [32,33], axis=1)
y_test = test[:,33]
test_oracle = test[:,32]

#add some basic features
df_lst = [x_train, x_val, x_test]
for i in range(len(df_lst)):
	feat_sum = df_lst[i].sum(1)[...,None]
	feat_mean = df_lst[i].mean(1)[...,None]
	feat_min = df_lst[i].min(1)[...,None]
	feat_max = df_lst[i].max(1)[...,None]
	feat_lst = [feat_sum, feat_mean, feat_min, feat_max]
	for feat in feat_lst:
		df_lst[i] = np.hstack((df_lst[i],feat))
		df_lst[i].shape

#run some preliminary models

#baseline model
x_train_exp = df_lst[0]
x_val_exp = df_lst[1]
x_test_exp = df_lst[2]
baseline_predicts = np.where(x_test_exp[:,32] > x_train_exp[:,32].mean(),1,0)
recall_score(y_test, baseline_predicts) #0.4892422825070159
precision_score(y_test, baseline_predicts) #0.0090043558356145519
accuracy_score(y_test, baseline_predicts) #0.83341599557350876

#Random Forest, no added features
clf_rf1 = RandomForestClassifier(n_estimators=25, random_state=0)
clf_rf1.fit(x_train, y_train) 
clf_rf1.score(x_val, y_val) #0.9994877279385993
clf_rf1.score(x_test, y_test) #0.99793296617919836
recall_score(y_test, clf_rf.predict(x_test)) #0.59588400374181483
precision_score(y_test, clf_rf.predict(x_test)) #0.68790496760259179

#Random Forest, with added features
clf_rf = RandomForestClassifier(n_estimators=25, random_state=0)
clf_rf.fit(x_train_exp, y_train) 
clf_rf.score(x_val_exp, y_val) 
clf_rf.score(x_test_exp, y_test) #0.99781255644206424
recall_score(y_test, clf_rf.predict(x_test_exp)) #0.6033676333021516
precision_score(y_test, clf_rf.predict(x_test_exp)) #0.65548780487804881
f1_score(y_test, clf_rf.predict(x_test_exp)) #0.6283487

#Oracle
x_train_orc = np.hstack((x_train_exp, train_oracle[...,None]))
x_val_orc = np.hstack((x_val_exp, val_oracle[...,None]))
x_test_orc = np.hstack((x_test_exp, test_oracle[...,None]))
rf_orc = RandomForestClassifier(n_estimators=25, random_state=0)
rf_orc.fit(x_train_orc, y_train)
accuracy_score(y_test, rf_orc.predict(x_test_orc)) #0.99756600317078969
precision_score(y_test, rf_orc.predict(x_test_orc)) #0.56124721603563477
recall_score(y_test, rf_orc.predict(x_test_orc)) #0.94293732460243218

#Logistic, added feats
clf_lg = LogisticRegression(random_state = 0)
clf_lg.fit(x_train_exp, y_train)
accuracy_score(y_test, clf_lg.predict(x_test_exp)) #0.75402297532460461
recall_score(y_test, clf_lg.predict(x_test_exp)) #0.78297474275023382
precision_score(y_test, clf_lg.predict(x_test_exp)) #0.0096870515253923428
f1_score(y_test, clf_lg.predict(x_test_exp)) #0.019137333805860093





