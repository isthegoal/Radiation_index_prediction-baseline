# coding=gbk
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from lightgbm import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn
import time
import random
import operator
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame as DF
start_time =time.time()
time_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

random.seed(1000)

'''
单模型+排名前80特征选择
'''
def my_score(estimator, X, y):
    predicted = estimator.predict(X,num_iteration=estimator.best_iteration)
    return mean_absolute_error(y,predicted)

def get_pic(model, feature_name):
    ans = DF()
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    #     print(ans[ans['score']>0].shape)
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)

def gey_100_impot_feature(train,y,nums=80):
    print('nums:',nums)
    #
    lgb_model = lgb.LGBMRegressor(n_estimators=120)
    # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
    lgb_model.fit(train, y)
    feature_name1 = train.columns
    #print('用到的特征：：：',feature_name.ix[:nums])
    return  list(set(get_pic(lgb_model,feature_name1).head(80)['name']))

def UseLightGBM():


    train = pd.read_csv( './data/extract_train_data.csv')
    test = pd.read_csv('./data/extract_test_data.csv')
    y = train.pop('label')

    features = [i for i in train.columns if i not in ['Unnamed: 0', 'label', '辐照度2', '辐照度5', '辐照度23']]
    train=train[features]
    test=test[features]

    print('train:',train.head())
    print('test:',test.head())
    ####################################    交叉检验   #####################################
    xx_cv = []
    xx_pre = []
    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=60)
    pred_cv = []
    for train_index, test_index in kf.split(train):
        X_train, X_test, y_train, y_test = train.values[train_index], train.values[test_index], y[train_index], y[test_index]
        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # specify your configurations as a dict
        params = {'metric': 'mae',  'boosting_type': 'gbdt',  'learning_rate': 0.01,
                 }

        print('Start training...')
        # train
        gbm = lgb.train(params,train_set=lgb_train,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=10,
                        verbose_eval=100,
                        )
        #将使用这个最好的得分作为xx_cv下的附加得分
        xx_cv.append(my_score(gbm,X_test,y_test))
        y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)
        pred_cv.append(y_pred)
        print('得分是：',xx_cv)
        xx_pre.append(gbm.predict(test))
    #获取最好cv得分的序号，使用这个序号去找到序号下对应的预测结果
    print('得分是：', xx_cv)
    print('平均得分是：', np.mean(xx_cv))
    ####################################    直接生成概率结果   #####################################
    res = np.array(pred_cv)
    r = res.mean(axis=0)
    res = pd.DataFrame()
    res['time'] = [i for i in range(1,916)]
    res['prediction'] = list(r)
    ####################################    提交和线下结果展示   #####################################
    res.to_csv(os.getcwd() +'/submit/%s_%s.csv'% (str(time_date), str(np.mean(xx_cv)).split('.')[1]),
               index=False, columns=['time','prediction'])
    print('线下成绩约', np.mean(xx_cv))


if __name__=='__main__':
    UseLightGBM()

'''
最基础特征
1.设置参数：
   'metric': 'mae','boosting_type': 'gbdt','learning_rate': 0.01,'max_depth':3,'num_leaves': 30,'colsample_bytree': 0.8
   线下0.1681，线上0.193   发生过拟合。
   真的是线下越小，线上越差
2.默认参数：
   'metric': 'mae',  'boosting_type': 'gbdt',   'learning_rate': 0.01,
   线下：0.169，线上：
3.调节参数：
   'metric': 'mae',  'boosting_type': 'gbdt',  'learning_rate': 0.01,  'n_estimators':100,
   'lambda':0.8,'max_depth':5,'num_leaves': 40, 'seed':2018
   线下：0.173
4.调节参数：
    'n_estimators'从100变成120
   线下：0.1714
'''

