import tpot
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train_new.csv')
test_data = pd.read_csv('test_new.csv')

y_train = train_data['TARGET']
X_train = train_data.drop(['TARGET'],axis=1)

y_test = test_data['TARGET']
X_test = test_data.drop(['TARGET'],axis=1)

X_train,X_validation,y_train,y_validation = train_test_split(X_t_v, y_t_v, test_size=0.25)

classifier_config_dict = {
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini"],
        'max_features': np.arange(0.50, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    }
}

tpot_clf=tpot.TPOTClassifier(generations=20, population_size=50,
                          offspring_size=80, mutation_rate=0.9,
                          crossover_rate=0.1,
                          scoring='roc_auc', cv=3,
                          subsample=1,
                          n_jobs=-1,
                          max_time_mins=None, max_eval_time_mins=5,
                          random_state=42, config_dict=classifier_config_dict ,
                          warm_start=False,
                          memory=None,
                          use_dask=False,
                          periodic_checkpoint_folder=None,
                          early_stop=20,
                          verbosity=2,
                          disable_update_check=False)
#X_train, X_test, y_train, y_test = train_test_split(data_X,data_target,train_size=0.75, test_size=0.25)

tpot_clf.fit(X_train,y_train)
print('Validation auc score:',tpot_clf.score(X_validation,y_validation))
print('test auc score:',tpot_clf.score(X_test,y_test))