import featuretools as ft
import pandas as pd
import lightgbm as lgb
import numpy as np
import warnings
import time
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
data_path='C:\\Users\\admin\\Downloads\\all\\'


#读取数据,为了方便看到模型效果，我们只选择部分数据
#application_test_data=pd.read_csv(data_path+'application_test.csv').reset_index(drop = True).loc[:2, :]
application_train_data=pd.read_csv(data_path+'application_train.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:20000, :]
bureau_data=pd.read_csv(data_path+'bureau.csv').sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop = True).loc[:20000, :]
bureau_balance_data=pd.read_csv(data_path+'bureau_balance.csv').sort_values('SK_ID_BUREAU').reset_index(drop = True).loc[:20000, :]
credit_card_balance_data=pd.read_csv(data_path+'credit_card_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:20000, :]
installments_payments_data=pd.read_csv(data_path+'installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:20000, :]
POS_CASH_balance_data=pd.read_csv(data_path+'POS_CASH_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:20000, :]
previous_application_data=pd.read_csv(data_path+'previous_application.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:20000, :]


app_train=application_train_data
bureau=bureau_data
bureau_balance=bureau_balance_data
credit=credit_card_balance_data
installments=installments_payments_data
cash=POS_CASH_balance_data
previous=previous_application_data



bureau1=bureau.copy()
bureau_balance1=bureau_balance.copy()
credit1=credit.copy()
installments1=installments.copy()
cash1=cash.copy()
previous1=previous.copy()
#del application_train_data,bureau_data,bureau_balance_data,credit_card_balance_data,installments_payments_data,POS_CASH_balance_data,previous_application_data


def data_split(data,alpha):
    train,test=train_test_split(data,train_size=alpha,random_state=42)
    return train,test


app_train_,app_test_=data_split(app_train,0.7)    



def feature_(app,bureau,previous,bureau_balance,cash,installments,credit):
    es = ft.EntitySet(id = 'clients')
    # Entities with a unique index
    es = es.entity_from_dataframe(entity_id = 'app', dataframe = app, index = 'SK_ID_CURR')
    es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')
    es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, index = 'SK_ID_PREV')

    # Entities that do not have a unique index
    es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe = bureau_balance, 
                              make_index = True, index = 'bureaubalance_index')

    es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash, 
                              make_index = True, index = 'cash_index')

    es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments,
                              make_index = True, index = 'installments_index')

    es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,
                              make_index = True, index = 'credit_index')

    # Relationship between app and bureau
    r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

    # Relationship between bureau and bureau balance
    r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

    # Relationship between current app and previous apps
    r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

    # Relationships between previous apps and cash, installments, and credit
    r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
    r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
    r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])
    # Add in the defined relationships
    es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])

    es = es.add_relationships([r_app_bureau, r_bureau_balance])
    # Print out the EntitySet
    #es
    """

    # List the primitives in a dataframe
    primitives = ft.list_primitives()
    pd.options.display.max_colwidth = 100
    primitives[primitives['type'] == 'aggregation'].head(10)
    primitives[primitives['type'] == 'transform'].head(10)
    """
    # Default primitives from featuretools
    default_agg_primitives =  ['sum', 'count', 'min', 'max', 'mean', 'mode']
    default_trans_primitives =  ['day', 'year', 'month']

    start=time.time()
    # DFS with list primitives
    feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'app',
                                       trans_primitives = default_trans_primitives,
                                       agg_primitives=default_agg_primitives, 
                                        max_depth = 1, features_only=False, verbose = True,n_jobs=1)
    end=time.time()
    #特征衍生运行时间
    long=end-start
    return feature_matrix,feature_names,long
feature_matrix1,feature_names1,long1=feature_(app_train_,bureau,previous,bureau_balance,cash,installments,credit)
feature_matrix1.to_csv('train_new.csv',index=False)
feature_matrix2,feature_names2,long2=feature_(app_test_,bureau1,previous1,bureau_balance1,cash1,installments1,credit1)
feature_matrix2.to_csv('test_new.csv',index=False) 

combine_data=feature_matrix1.append(feature_matrix2)



from featuretools import selection

# Remove features with only one unique value  and all nan
feature_matrix3 = selection.remove_low_information_features(combine_data)

print('Removed %d features' % (combine_data.shape[1]- feature_matrix3.shape[1]))

target = 'TARGET'
predictors = [
 'NAME_CONTRACT_TYPE',
 'CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY',
 'CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE',
 'NAME_TYPE_SUITE',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'REGION_POPULATION_RELATIVE',
 'DAYS_BIRTH',
 'DAYS_EMPLOYED',
 'DAYS_REGISTRATION',
 'DAYS_ID_PUBLISH',
 'OWN_CAR_AGE',
 'FLAG_MOBIL',
 'FLAG_EMP_PHONE',
 'FLAG_WORK_PHONE',
 'FLAG_CONT_MOBILE',
 'FLAG_PHONE',
 'FLAG_EMAIL',
 'OCCUPATION_TYPE',
 'CNT_FAM_MEMBERS',
 'REGION_RATING_CLIENT',
 'REGION_RATING_CLIENT_W_CITY',
 'WEEKDAY_APPR_PROCESS_START',
 'HOUR_APPR_PROCESS_START',
 'REG_REGION_NOT_LIVE_REGION',
 'REG_REGION_NOT_WORK_REGION',
 'LIVE_REGION_NOT_WORK_REGION',
 'REG_CITY_NOT_LIVE_CITY',
 'REG_CITY_NOT_WORK_CITY',
 'LIVE_CITY_NOT_WORK_CITY',
 'ORGANIZATION_TYPE',
 'EXT_SOURCE_1',
 'EXT_SOURCE_2',
 'EXT_SOURCE_3',
 'APARTMENTS_AVG',
 'BASEMENTAREA_AVG',
 'YEARS_BEGINEXPLUATATION_AVG',
 'YEARS_BUILD_AVG',
 'COMMONAREA_AVG',
 'ELEVATORS_AVG',
 'ENTRANCES_AVG',
 'FLOORSMAX_AVG',
 'FLOORSMIN_AVG',
 'LANDAREA_AVG',
 'LIVINGAPARTMENTS_AVG',
 'LIVINGAREA_AVG',
 'NONLIVINGAPARTMENTS_AVG',
 'NONLIVINGAREA_AVG',
 'APARTMENTS_MODE',
 'BASEMENTAREA_MODE',
 'YEARS_BEGINEXPLUATATION_MODE',
 'YEARS_BUILD_MODE',
 'COMMONAREA_MODE',
 'ELEVATORS_MODE',
 'ENTRANCES_MODE',
 'FLOORSMAX_MODE',
 'FLOORSMIN_MODE',
 'LANDAREA_MODE',
 'LIVINGAPARTMENTS_MODE',
 'LIVINGAREA_MODE',
 'NONLIVINGAPARTMENTS_MODE',
 'NONLIVINGAREA_MODE',
 'APARTMENTS_MEDI',
 'BASEMENTAREA_MEDI',
 'YEARS_BEGINEXPLUATATION_MEDI',
 'YEARS_BUILD_MEDI',
 'COMMONAREA_MEDI',
 'ELEVATORS_MEDI',
 'ENTRANCES_MEDI',
 'FLOORSMAX_MEDI',
 'FLOORSMIN_MEDI',
 'LANDAREA_MEDI',
 'LIVINGAPARTMENTS_MEDI',
 'LIVINGAREA_MEDI',
 'NONLIVINGAPARTMENTS_MEDI',
 'NONLIVINGAREA_MEDI',
 'FONDKAPREMONT_MODE',
 'HOUSETYPE_MODE',
 'TOTALAREA_MODE',
 'WALLSMATERIAL_MODE',
 'EMERGENCYSTATE_MODE',
 'OBS_30_CNT_SOCIAL_CIRCLE',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'DEF_60_CNT_SOCIAL_CIRCLE',
 'DAYS_LAST_PHONE_CHANGE',
 'FLAG_DOCUMENT_3',
 'FLAG_DOCUMENT_4',
 'FLAG_DOCUMENT_5',
 'FLAG_DOCUMENT_6',
 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_8',
 'FLAG_DOCUMENT_9',
 'FLAG_DOCUMENT_11',
 'FLAG_DOCUMENT_13',
 'FLAG_DOCUMENT_14',
 'FLAG_DOCUMENT_15',
 'FLAG_DOCUMENT_16',
 'FLAG_DOCUMENT_17',
 'FLAG_DOCUMENT_18',
 'FLAG_DOCUMENT_19',
 'FLAG_DOCUMENT_20',
 'FLAG_DOCUMENT_21',
 'AMT_REQ_CREDIT_BUREAU_HOUR',
 'AMT_REQ_CREDIT_BUREAU_DAY',
 'AMT_REQ_CREDIT_BUREAU_WEEK',
 'AMT_REQ_CREDIT_BUREAU_MON',
 'AMT_REQ_CREDIT_BUREAU_QRT',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'SUM(previous.AMT_ANNUITY)',
 'SUM(previous.AMT_APPLICATION)',
 'SUM(previous.AMT_CREDIT)',
 'SUM(previous.AMT_DOWN_PAYMENT)',
 'SUM(previous.AMT_GOODS_PRICE)',
 'SUM(previous.HOUR_APPR_PROCESS_START)',
 'SUM(previous.NFLAG_LAST_APPL_IN_DAY)',
 'SUM(previous.RATE_DOWN_PAYMENT)',
 'SUM(previous.RATE_INTEREST_PRIMARY)',
 'SUM(previous.RATE_INTEREST_PRIVILEGED)',
 'SUM(previous.DAYS_DECISION)',
 'SUM(previous.SELLERPLACE_AREA)',
 'SUM(previous.CNT_PAYMENT)',
 'SUM(previous.DAYS_FIRST_DRAWING)',
 'SUM(previous.DAYS_FIRST_DUE)',
 'SUM(previous.DAYS_LAST_DUE_1ST_VERSION)',
 'SUM(previous.DAYS_LAST_DUE)',
 'SUM(previous.DAYS_TERMINATION)',
 'SUM(previous.NFLAG_INSURED_ON_APPROVAL)',
 'COUNT(previous)',
 'MIN(previous.AMT_ANNUITY)',
 'MIN(previous.AMT_APPLICATION)',
 'MIN(previous.AMT_CREDIT)',
 'MIN(previous.AMT_DOWN_PAYMENT)',
 'MIN(previous.AMT_GOODS_PRICE)',
 'MIN(previous.HOUR_APPR_PROCESS_START)',
 'MIN(previous.NFLAG_LAST_APPL_IN_DAY)',
 'MIN(previous.RATE_DOWN_PAYMENT)',
 'MIN(previous.RATE_INTEREST_PRIMARY)',
 'MIN(previous.RATE_INTEREST_PRIVILEGED)',
 'MIN(previous.DAYS_DECISION)',
 'MIN(previous.SELLERPLACE_AREA)',
 'MIN(previous.CNT_PAYMENT)',
 'MIN(previous.DAYS_FIRST_DRAWING)',
 'MIN(previous.DAYS_FIRST_DUE)',
 'MIN(previous.DAYS_LAST_DUE_1ST_VERSION)',
 'MIN(previous.DAYS_LAST_DUE)',
 'MIN(previous.DAYS_TERMINATION)',
 'MIN(previous.NFLAG_INSURED_ON_APPROVAL)',
 'MAX(previous.AMT_ANNUITY)',
 'MAX(previous.AMT_APPLICATION)',
 'MAX(previous.AMT_CREDIT)',
 'MAX(previous.AMT_DOWN_PAYMENT)',
 'MAX(previous.AMT_GOODS_PRICE)',
 'MAX(previous.HOUR_APPR_PROCESS_START)',
 'MAX(previous.NFLAG_LAST_APPL_IN_DAY)',
 'MAX(previous.RATE_DOWN_PAYMENT)',
 'MAX(previous.RATE_INTEREST_PRIMARY)',
 'MAX(previous.RATE_INTEREST_PRIVILEGED)',
 'MAX(previous.DAYS_DECISION)',
 'MAX(previous.SELLERPLACE_AREA)',
 'MAX(previous.CNT_PAYMENT)',
 'MAX(previous.DAYS_FIRST_DRAWING)',
 'MAX(previous.DAYS_FIRST_DUE)',
 'MAX(previous.DAYS_LAST_DUE_1ST_VERSION)',
 'MAX(previous.DAYS_LAST_DUE)',
 'MAX(previous.DAYS_TERMINATION)',
 'MAX(previous.NFLAG_INSURED_ON_APPROVAL)',
 'MEAN(previous.AMT_ANNUITY)',
 'MEAN(previous.AMT_APPLICATION)',
 'MEAN(previous.AMT_CREDIT)',
 'MEAN(previous.AMT_DOWN_PAYMENT)',
 'MEAN(previous.AMT_GOODS_PRICE)',
 'MEAN(previous.HOUR_APPR_PROCESS_START)',
 'MEAN(previous.NFLAG_LAST_APPL_IN_DAY)',
 'MEAN(previous.RATE_DOWN_PAYMENT)',
 'MEAN(previous.RATE_INTEREST_PRIMARY)',
 'MEAN(previous.RATE_INTEREST_PRIVILEGED)',
 'MEAN(previous.DAYS_DECISION)',
 'MEAN(previous.SELLERPLACE_AREA)',
 'MEAN(previous.CNT_PAYMENT)',
 'MEAN(previous.DAYS_FIRST_DRAWING)',
 'MEAN(previous.DAYS_FIRST_DUE)',
 'MEAN(previous.DAYS_LAST_DUE_1ST_VERSION)',
 'MEAN(previous.DAYS_LAST_DUE)',
 'MEAN(previous.DAYS_TERMINATION)',
 'MEAN(previous.NFLAG_INSURED_ON_APPROVAL)',
 'MODE(previous.NAME_CONTRACT_TYPE)',
 'MODE(previous.WEEKDAY_APPR_PROCESS_START)',
 'MODE(previous.FLAG_LAST_APPL_PER_CONTRACT)',
 'MODE(previous.NAME_CASH_LOAN_PURPOSE)',
 'MODE(previous.NAME_CONTRACT_STATUS)',
 'MODE(previous.NAME_PAYMENT_TYPE)',
 'MODE(previous.CODE_REJECT_REASON)',
 'MODE(previous.NAME_TYPE_SUITE)',
 'MODE(previous.NAME_CLIENT_TYPE)',
 'MODE(previous.NAME_GOODS_CATEGORY)',
 'MODE(previous.NAME_PORTFOLIO)',
 'MODE(previous.NAME_PRODUCT_TYPE)',
 'MODE(previous.CHANNEL_TYPE)',
 'MODE(previous.NAME_SELLER_INDUSTRY)',
 'MODE(previous.NAME_YIELD_GROUP)',
 'MODE(previous.PRODUCT_COMBINATION)',
 'SUM(bureau.DAYS_CREDIT)',
 'SUM(bureau.CREDIT_DAY_OVERDUE)',
 'SUM(bureau.DAYS_CREDIT_ENDDATE)',
 'SUM(bureau.DAYS_ENDDATE_FACT)',
 'SUM(bureau.AMT_CREDIT_MAX_OVERDUE)',
 'SUM(bureau.CNT_CREDIT_PROLONG)',
 'SUM(bureau.AMT_CREDIT_SUM)',
 'SUM(bureau.AMT_CREDIT_SUM_DEBT)',
 'SUM(bureau.AMT_CREDIT_SUM_LIMIT)',
 'SUM(bureau.AMT_CREDIT_SUM_OVERDUE)',
 'SUM(bureau.DAYS_CREDIT_UPDATE)',
 'SUM(bureau.AMT_ANNUITY)',
 'COUNT(bureau)',
 'MIN(bureau.DAYS_CREDIT)',
 'MIN(bureau.CREDIT_DAY_OVERDUE)',
 'MIN(bureau.DAYS_CREDIT_ENDDATE)',
 'MIN(bureau.DAYS_ENDDATE_FACT)',
 'MIN(bureau.AMT_CREDIT_MAX_OVERDUE)',
 'MIN(bureau.CNT_CREDIT_PROLONG)',
 'MIN(bureau.AMT_CREDIT_SUM)',
 'MIN(bureau.AMT_CREDIT_SUM_DEBT)',
 'MIN(bureau.AMT_CREDIT_SUM_LIMIT)',
 'MIN(bureau.AMT_CREDIT_SUM_OVERDUE)',
 'MIN(bureau.DAYS_CREDIT_UPDATE)',
 'MIN(bureau.AMT_ANNUITY)',
 'MAX(bureau.DAYS_CREDIT)',
 'MAX(bureau.CREDIT_DAY_OVERDUE)',
 'MAX(bureau.DAYS_CREDIT_ENDDATE)',
 'MAX(bureau.DAYS_ENDDATE_FACT)',
 'MAX(bureau.AMT_CREDIT_MAX_OVERDUE)',
 'MAX(bureau.CNT_CREDIT_PROLONG)',
 'MAX(bureau.AMT_CREDIT_SUM)',
 'MAX(bureau.AMT_CREDIT_SUM_DEBT)',
 'MAX(bureau.AMT_CREDIT_SUM_LIMIT)',
 'MAX(bureau.AMT_CREDIT_SUM_OVERDUE)',
 'MAX(bureau.DAYS_CREDIT_UPDATE)',
 'MAX(bureau.AMT_ANNUITY)',
 'MEAN(bureau.DAYS_CREDIT)',
 'MEAN(bureau.CREDIT_DAY_OVERDUE)',
 'MEAN(bureau.DAYS_CREDIT_ENDDATE)',
 'MEAN(bureau.DAYS_ENDDATE_FACT)',
 'MEAN(bureau.AMT_CREDIT_MAX_OVERDUE)',
 'MEAN(bureau.CNT_CREDIT_PROLONG)',
 'MEAN(bureau.AMT_CREDIT_SUM)',
 'MEAN(bureau.AMT_CREDIT_SUM_DEBT)',
 'MEAN(bureau.AMT_CREDIT_SUM_LIMIT)',
 'MEAN(bureau.AMT_CREDIT_SUM_OVERDUE)',
 'MEAN(bureau.DAYS_CREDIT_UPDATE)',
 'MEAN(bureau.AMT_ANNUITY)',
 'MODE(bureau.CREDIT_ACTIVE)',
 'MODE(bureau.CREDIT_CURRENCY)',
 'MODE(bureau.CREDIT_TYPE)']
categorical = ['NAME_CONTRACT_TYPE',
 'CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY',
 'NAME_TYPE_SUITE',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'OCCUPATION_TYPE',
 'WEEKDAY_APPR_PROCESS_START',
 'ORGANIZATION_TYPE',
 'FONDKAPREMONT_MODE',
 'HOUSETYPE_MODE',
 'WALLSMATERIAL_MODE',
 'EMERGENCYSTATE_MODE',
 'MODE(previous.NAME_CONTRACT_TYPE)',
 'MODE(previous.WEEKDAY_APPR_PROCESS_START)',
 'MODE(previous.FLAG_LAST_APPL_PER_CONTRACT)',
 'MODE(previous.NAME_CASH_LOAN_PURPOSE)',
 'MODE(previous.NAME_CONTRACT_STATUS)',
 'MODE(previous.NAME_PAYMENT_TYPE)',
 'MODE(previous.CODE_REJECT_REASON)',
 'MODE(previous.NAME_TYPE_SUITE)',
 'MODE(previous.NAME_CLIENT_TYPE)',
 'MODE(previous.NAME_GOODS_CATEGORY)',
 'MODE(previous.NAME_PORTFOLIO)',
 'MODE(previous.NAME_PRODUCT_TYPE)',
 'MODE(previous.CHANNEL_TYPE)',
 'MODE(previous.NAME_SELLER_INDUSTRY)',
 'MODE(previous.NAME_YIELD_GROUP)',
 'MODE(previous.PRODUCT_COMBINATION)',
 'MODE(bureau.CREDIT_ACTIVE)',
 'MODE(bureau.CREDIT_CURRENCY)',
 'MODE(bureau.CREDIT_TYPE)']

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in categorical:
    feature_matrix3[col] = label_encoder.fit_transform(np.array(feature_matrix3[col].astype(str)).reshape((-1,)))

feature_matrix3.reset_index(drop=True,inplace=True)
train = feature_matrix3.ix[:13999,:]
test = feature_matrix3.ix[14000:,:].reset_index(drop=True)
print('Final Training Shape: ', train.shape)
print('Final Testing Shape: ', test.shape)

def under_sample(data):
    na_len=len(data.ix[data.TARGET==1,0])
    train_n,train2=train_test_split(data.ix[data.TARGET==0,:],train_size=2*na_len,random_state=42)
    train_p=data.ix[data.TARGET==1,:]
    return train_p,train_n
train_p,train_n=under_sample(train)
train_new=train_p.append(train_n,ignore_index=True)
test_p,test_n=under_sample(test)
test_new=test_p.append(test_n,ignore_index=True)



#调参
from hyperopt import STATUS_OK

N_FOLDS = 5

# Create the dataset
train_set = lgb.Dataset(train_new.drop(['TARGET'],axis=1), train_new.TARGET)


def objective(params, n_folds=N_FOLDS):
   '''Objective function for Gradient Boosting Machine Hyperparameter Tuning'''

   # Perform n_fold cross validation with hyperparameters
   # Use early stopping and evalute based on ROC AUC
   #这里最好加上类别型特征
   cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=100,
                       early_stopping_rounds=200, metrics='auc', seed=50)

   # Extract the best score
   best_score = max(cv_results['auc-mean'])

   # Loss must be minimized
   loss = 1 - best_score

   # Dictionary with information for evaluation
   return {'loss': loss, 'params': params, 'status': STATUS_OK}
from hyperopt import hp
# Define the search space
space = {'boosting_type': 'gbdt', # 训练方式
         'objective': 'binary', # 目标 二分类
         'metric': 'auc', # 损失函数
         'is_unbalance': True,
         'subsample_for_bin':hp.choice("max_depth",list(range(50000,200000,10000))),
         'max_depth':hp.choice("max_depth",list(range(3,10,1))),
         'num_leaves':hp.choice('num_leaves',list(range(20,200,5))),
         'feature_fraction':hp.loguniform('feature_fraction',np.log(0.1), np.log(1)),
         'bagging_fraction':hp.loguniform('bagging_fraction',np.log(0.1), np.log(1)),
         'bagging_freq':hp.choice('bagging_freq',range(20,60,5)),
         'max_bin':hp.choice('max_bin',list(range(10,255,10))),
         'min_data_in_leaf': hp.choice('min_data_in_leaf',list(range(10,200,5))),
         'min_sum_hessian_in_leaf':hp.loguniform('min_sum_hessian_in_leaf', np.log(0.0001), np.log(0.005)),
         'lambda_l1': hp.uniform('lambda_l1', 0,1),
         'lambda_l2': hp.uniform('lambda_l2', 0,1),
         'min_split_gain':hp.loguniform( 'min_split_gain' ,np.log(0.1), np.log(1)),
         'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2))
         }
from hyperopt import tpe
# Algorithm
tpe_algorithm = tpe.suggest
from hyperopt import fmin
from hyperopt import Trials
# Trials object to track progress
bayes_trials = Trials()
MAX_EVALS = 500
# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
           max_evals = MAX_EVALS, trials = bayes_trials)


model = lgb.LGBMClassifier(
        boosting_type='gbdt', # 训练方式
        n_estimators=100,
        objective='binary', # 目标 二分类
        metric='auc', # 损失函数
        is_unbalance=True,
        bagging_fraction=0.4039339074985622,
        bagging_freq=5,
        feature_fraction=0.21615529449498946,
        lambda_l1=5,
        lambda_l2=4,
        subsample_for_bin=200000,
        learning_rate=0.011658729544476418,
        max_bin=2,
        max_depth=5,
        min_data_in_leaf=29,
        min_split_gain=0.7448866777842496,
        min_sum_hessian_in_leaf=0.002638713430308378,
        num_leaves=24,
        nthread=8,
        verbose=1,
        tree_learner='data')
model.fit(train_new.drop(['TARGET'],axis=1), train_new.TARGET)
p=model.predict(train_new.drop(['TARGET'],axis=1))
auc=roc_auc_score(test_new.TARGET,p)
