import numpy as np 
import pandas as pd 
import autosklearn
import autosklearn.classification
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

org_data = pd.read_csv('app_org.csv')
org_data = org_data.drop(['SK_ID_CURR'],axis=1)


'''
#============================================
#random forest'
#============================================
def fillna(col):
	col.fillna(col.value_counts().index[0],inplace=True)
	return col

org_data = org_data.apply(lambda col:fillna(col))
y = org_data['TARGET']
X = org_data.drop(['TARGET'],axis=1)
X_t_v, X_test, y_t_v, y_test = train_test_split(X,y,test_size=0.25)

X_train,X_validation,y_train,y_validation = train_test_split(X_t_v,y_t_v,test_size=0.25)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_vhat = clf.predict(X_validation)
y_hat = clf.predict(X_test)
print('validation auc score:',sklearn.metrics.roc_auc_score(y_validation,y_vhat))
print('test auc score:',sklearn.metrics.roc_auc_score(y_test,y_hat))
'''



#=========================================
#AutoSklearn
#=========================================
y = org_data['TARGET']
X = org_data.drop(['TARGET'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=540,
		per_run_time_limit=90,ml_memory_limit=10240)
automl.fit(X_train,y_train)
#print(automl.show_models())
y_hat = automl.predict(X_test)
print(automl.sprint_statistics())
print('auc score:',sklearn.metrics.roc_auc_score(y_test,y_hat))

