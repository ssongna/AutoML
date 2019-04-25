import pandas as pd 
import autosklearn
import autosklearn.classification
import numpy as np
import sklearn.metrics
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


imp = Imputer(missing_values='NaN',strategy='most_frequent')

#no 'object' type features
#all the features are encoded 

'''
(3333,)
(3333, 262)
(1401,)
(1401, 262)
'''
train_data = pd.read_csv('train_new.csv')
test_data = pd.read_csv('test_new.csv')


y_train = train_data['TARGET']
X_train = train_data.drop(['TARGET'],axis=1)


y_test = test_data['TARGET']
X_test = test_data.drop(['TARGET'],axis=1)



'''
#============================================
#random forest
#============================================
def fillna(col):
	col.fillna(col.value_counts().index[0],inplace=True)
	return col

for item in [train_data,test_data]:
	item = item.apply(lambda col:fillna(col))


clf = RandomForestClassifier()
#clf = svm.SVC(C=10.0)
clf.fit(X_train,y_train)
y_vhat = clf.predict(X_validation)
y_hat = clf.predict(X_test)
print('Validation Auc Score:',sklearn.metrics.roc_auc_score(y_validation,y_vhat))
print('Test Auc score:',sklearn.metrics.roc_auc_score(y_test,y_hat))
'''



#=========================================
#AutoSklearn
#=========================================

#include_preprocessors=['no_preprocessing']

automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=180,
															per_run_time_limit=30,
															ml_memory_limit=10240,
															ensemble_size=3,
															ensemble_nbest=3)
automl.fit(X_train,y_train)

y_vhat = automl.predict(X_validation)
y_hat = automl.predict(X_test)
print('get model with weight:')
print(automl.get_models_with_weights())
print('get parameters:')
print(automl.get_params(deep=True))
print('show models:')
print(automl.show_models())
print('sprint_statistics()')
print(automl.sprint_statistics())

print('Validation Auc Score:',sklearn.metrics.roc_auc_score(y_validation,y_vhat))
print('Test Auc score:',sklearn.metrics.roc_auc_score(y_test,y_hat))

#dif = np.sum(np.absolute(np.array(y_hat) - np.array(y_test)))
#print("persentage of all dataset:", (1.0-dif*1.0/(len(y_test)*1.0)))

