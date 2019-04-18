# coding=gbk
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import time
import h5py


def ori_data_flat():
    #打开文件
    train = pd.read_csv( '../data/train_feature.csv')
    train_label = pd.read_csv('../data/train_label.csv')
    test = pd.read_csv( '../data/test_feature.csv')
    print('--------  对训练集进行时间信息的时序平展  -----------')
    the_big_list=[]
    def day_data(i):
        small_list=[]
        i=DataFrame(i)
        i.reset_index(drop=True)
        i.drop(['日期', '时刻'], axis=1,inplace=True)
        for j in range(0,8):
            small_list.extend(list(i.iloc[j]))
        print('oooo',list(i.iloc[j]))
        the_big_list.append(small_list)
        print(i)
    train.groupby('日期').apply(day_data)
    #提取完成，进行放置
    print(the_big_list)
    the_column_list=[]
    for i in ['2','5','8','11','14','17','20','23']:
        for j in ['辐照度','风速','风向','温度','湿度','气压']:
            the_column_list.append(j+i)
    the_train=pd.DataFrame(the_big_list,columns=the_column_list)
    print(the_train.head())
    #多了一行  删去第一行
    the_train.drop([0],inplace=True)
    #加上标签列
    print('label:',train_label['电场实际太阳辐射指数'])
    the_train['label']=list(train_label['电场实际太阳辐射指数'])
    the_train.to_csv('../data/extract_train_data.csv')
    #
    print('--------  对测试集进行时间信息的时序平展  -----------')
    the_big_list=[]
    def day_data(i):
        small_list=[]
        i=DataFrame(i)
        i.reset_index(drop=True)
        i.drop(['日期', '时刻'], axis=1,inplace=True)
        for j in range(0,8):
            small_list.extend(list(i.iloc[j]))
        print('oooo',list(i.iloc[j]))
        the_big_list.append(small_list)
        print(i)
    test.groupby('日期').apply(day_data)
    #提取完成，进行放置
    #print(train.head())
    print(the_big_list)
    the_column_list=[]
    for i in ['2','5','8','11','14','17','20','23']:
        for j in ['辐照度','风速','风向','温度','湿度','气压']:
            the_column_list.append(j+i)
    the_test=pd.DataFrame(the_big_list,columns=the_column_list)
    #多了一行  删去第一行
    the_test.drop([0],inplace=True)
    print(the_test)
    the_test.to_csv('../data/extract_test_data.csv')
    pass
if __name__=="__main__":
    print('对初始数据按照时序的平展')
    ori_data_flat()