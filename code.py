#huangzy97
#####导入需要的包
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import time
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')
#####导入数据
test1  = open('/mnt/sd04/sjjs_js06/Contest1_Training_1.txt')
test2  = open('/mnt/sd04/sjjs_js06/Contest1_Training_2.txt')
test6  = open('/mnt/sd04/sjjs_js06/Contest1_Training_6.txt')
fore1  = open('/mnt/sd04/sjjs_js06/Contest1_Forecast_1.txt')
fore2  = open('/mnt/sd04/sjjs_js06/Contest1_Forecast_2.txt')
fore6  = open('/mnt/sd04/sjjs_js06/Contest1_Forecast_6.txt')
t1 = pd.read_csv(test1,sep='|',header=None)
t1.columns = ['MONTH_ID', 'CUST_ID', 'USER_ID', 'USER_STATUS', 'AGE', 'SEX', 'IS_VALID_FLAG', 'FALSE_FLAG']
t2 = pd.read_csv(test2,sep='|',header=None)
t2.columns = ['MONTH_ID', 'CUST_ID', 'CHNL_ID', 'CHNL_KIND_ID', 'AREA_ID', 'PAY_MODE', 'CUST_SEX', 'CERT_AGE', 'CONSTELLATION_DESC']
t6 = pd.read_csv(test6,sep='|',header=None)
t6.columns = ['MONTH_ID','USER_ID','TOTAL_DURA','TOLL_DURA','ROAM_DURA','TOTAL_TIMES','TOTAL_L_TIMES','TOTAL_NUMS', 'OUT_CNT']
f1 = pd.read_csv(fore1,sep='|',header=None)
f1.columns = ['MONTH_ID', 'CUST_ID', 'USER_ID', 'USER_STATUS', 'AGE', 'SEX', 'IS_VALID_FLAG']
f2 = pd.read_csv(fore2,sep='|',header=None)
f2.columns = ['MONTH_ID', 'CUST_ID', 'CHNL_ID', 'CHNL_KIND_ID', 'AREA_ID', 'PAY_MODE', 'CUST_SEX', 'CERT_AGE', 'CONSTELLATION_DESC']
f6 = pd.read_csv(fore6,sep='|',header=None)
f6.columns = ['MONTH_ID','USER_ID','TOTAL_DURA','TOLL_DURA','ROAM_DURA','TOTAL_TIMES','TOTAL_L_TIMES','TOTAL_NUMS', 'OUT_CNT']
#########关联表
train = pd.merge(t1, t2, on='CUST_ID',how = 'outer')
#train.rename(columns={'DEVICE_NUMBER_x':'DEVICE_NUMBER'}, inplace = True)
#train = pd.merge(train, t3, on='DEVICE_NUMBER')
test  = pd.merge(f1, f2, on='CUST_ID',how = 'outer')
#test.rename(columns={'DEVICE_NUMBER_x':'DEVICE_NUMBER'}, inplace = True)
#####处理每个表的字段，清洗数据
###########t6
t6_01 = t6[t6['MONTH_ID']==201801]
t6_01.columns = ['MONTH_ID_01','USER_ID','TOTAL_DURA_01','TOLL_DURA_01','ROAM_DURA_01','TOTAL_TIMES_01','TOTAL_L_TIMES_01','TOTAL_NUMS_01', 'OUT_CNT_01']
del t6_01['MONTH_ID_01']
t6_02 = t6[t6['MONTH_ID']==201802]
t6_02.columns = ['MONTH_ID_02','USER_ID','TOTAL_DURA_02','TOLL_DURA_02','ROAM_DURA_02','TOTAL_TIMES_02','TOTAL_L_TIMES_02','TOTAL_NUMS_02', 'OUT_CNT_02']
del t6_02['MONTH_ID_02']
t6_12 = t6[t6['MONTH_ID']==201712]
t6_12.columns = ['MONTH_ID_12','USER_ID','TOTAL_DURA_12','TOLL_DURA_12','ROAM_DURA_12','TOTAL_TIMES_12','TOTAL_L_TIMES_12','TOTAL_NUMS_12', 'OUT_CNT_12']
del t6_12['MONTH_ID_12']
t6_re = pd.merge(t6_02,t6_01,on = 'USER_ID',how = 'outer')
t6_re = pd.merge(t6_re,t6_12,on = 'USER_ID',how = 'outer')
###########f6
f6_01 = f6[f6['MONTH_ID']==201801]
f6_01.columns = ['MONTH_ID_01','USER_ID','TOTAL_DURA_01','TOLL_DURA_01','ROAM_DURA_01','TOTAL_TIMES_01','TOTAL_L_TIMES_01','TOTAL_NUMS_01', 'OUT_CNT_01']
del f6_01['MONTH_ID_01']
f6_02 = f6[f6['MONTH_ID']==201802]
f6_02.columns = ['MONTH_ID_02','USER_ID','TOTAL_DURA_02','TOLL_DURA_02','ROAM_DURA_02','TOTAL_TIMES_02','TOTAL_L_TIMES_02','TOTAL_NUMS_02', 'OUT_CNT_02']
del f6_02['MONTH_ID_02']
f6_12 = f6[f6['MONTH_ID']==201712]
f6_12.columns = ['MONTH_ID_12','USER_ID','TOTAL_DURA_12','TOLL_DURA_12','ROAM_DURA_12','TOTAL_TIMES_12','TOTAL_L_TIMES_12','TOTAL_NUMS_12', 'OUT_CNT_12']
del f6_12['MONTH_ID_12']
f6_re = pd.merge(f6_02,f6_01,on = 'USER_ID',how = 'outer')
f6_re = pd.merge(f6_re,f6_12,on = 'USER_ID',how = 'outer')
#print(t4.MONTH_ID.value_counts())
#print(f6_re.columns)
#print(t6.MONTH_ID.value_counts())
#print(f6_re.shape)
#print(f6_re.head(30))
####构造新的宽表
test  = pd.merge(test, f6_re, on='USER_ID', how='outer')
####构造新的宽表
train = pd.merge(train, t6_re, on='USER_ID', how='outer')
#print(train.columns)
#print(test.columns)
label = [x for x in train.columns if x not in ['USER_ID', 'FALSE_FLAG','CHNL_ID','TOTAL_NUMS_01','TOTAL_NUMS_02','TOTAL_NUMS_12','OUT_CNT_12','OUT_CNT_01','OUT_CNT_02']]
train.drop(label,axis=1,inplace=True)
label2 = [x for x in test.columns if x not in ['USER_ID','CHNL_ID','TOTAL_NUMS_01','TOTAL_NUMS_02','TOTAL_NUMS_12','OUT_CNT_12','OUT_CNT_01','OUT_CNT_02']]
test.drop(label2,axis=1,inplace=True)
#print(train.head(10))
#print(test.head(20))
####合并train和test数据
train['source']= 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)
print(data.apply(lambda x: sum(x.isnull())))
print(data.dtypes)
data['TOTAL_NUMS_12'] = data['TOTAL_NUMS_12'].fillna(0)
data['TOTAL_NUMS_01'] = data['TOTAL_NUMS_01'].fillna(0)
data['OUT_CNT_12'] = data['OUT_CNT_12'].fillna(0)
data['OUT_CNT_01'] = data['OUT_CNT_01'].fillna(0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['CHNL_ID']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col].astype(str))
data = pd.get_dummies(data, columns=var_to_encode)
#####拆分train和test
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']
train.drop('source',axis=1,inplace=True)
test.drop(['source','FALSE_FLAG'],axis=1,inplace=True)

print(train.head(10))
print(test.head(20))

######模型
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
target='FALSE_FLAG'
IDcol = 'USER_ID'
######测试模型建立
def modelfit(alg, dtrain, dtest, predictors1,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors1].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors1].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print (cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors1], dtrain['FALSE_FLAG'],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors1])
    print(dtrain_predictions)
    dtrain_predprob = alg.predict_proba(dtrain[predictors1])[:,1]
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['FALSE_FLAG'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['FALSE_FLAG'], dtrain_predprob))
#####预测
    dtest_predictions = alg.predict(dtest[predictors])
#    test_y = alg.predict(xgtest)
    result = test
    result['FALSE_FLAG'] = dtest_predictions
    result['FALSE_FLAG'] = result['FALSE_FLAG'].map(lambda x: 1 if x>= 0.9 else 0)
    result[['USER_ID','FALSE_FLAG']].to_csv('/mnt/sd04/sjjs_js06/1_江苏6队.csv',index=False,header=False)
    xgb_fea_imp=pd.DataFrame(list(alg.get_booster().get_fscore().items()),columns=['feature','importance']).sort_values('importance', ascending=False)
    print('',xgb_fea_imp)
    xgb_fea_imp.to_csv('/mnt/sd04/sjjs_js06/xgb_fea_imp.csv')
predictors = [x for x in train.columns if x not in [target, IDcol]]
#46078
####最终模型的参数
xgb1 = XGBClassifier(
        learning_rate =0.01,
        n_estimators=4000,
        max_depth=4,
        min_child_weight=3,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
print(modelfit(xgb1, train, test, predictors))
