# coding=gbk
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import time
import h5py


def ori_data_flat():
    #���ļ�
    train = pd.read_csv( '../data/train_feature.csv')
    train_label = pd.read_csv('../data/train_label.csv')
    test = pd.read_csv( '../data/test_feature.csv')
    print('--------  ��ѵ��������ʱ����Ϣ��ʱ��ƽչ  -----------')
    the_big_list=[]
    def day_data(i):
        small_list=[]
        i=DataFrame(i)
        i.reset_index(drop=True)
        i.drop(['����', 'ʱ��'], axis=1,inplace=True)
        for j in range(0,8):
            small_list.extend(list(i.iloc[j]))
        print('oooo',list(i.iloc[j]))
        the_big_list.append(small_list)
        print(i)
    train.groupby('����').apply(day_data)
    #��ȡ��ɣ����з���
    print(the_big_list)
    the_column_list=[]
    for i in ['2','5','8','11','14','17','20','23']:
        for j in ['���ն�','����','����','�¶�','ʪ��','��ѹ']:
            the_column_list.append(j+i)
    the_train=pd.DataFrame(the_big_list,columns=the_column_list)
    print(the_train.head())
    #����һ��  ɾȥ��һ��
    the_train.drop([0],inplace=True)
    #���ϱ�ǩ��
    print('label:',train_label['�糡ʵ��̫������ָ��'])
    the_train['label']=list(train_label['�糡ʵ��̫������ָ��'])
    the_train.to_csv('../data/extract_train_data.csv')
    #
    print('--------  �Բ��Լ�����ʱ����Ϣ��ʱ��ƽչ  -----------')
    the_big_list=[]
    def day_data(i):
        small_list=[]
        i=DataFrame(i)
        i.reset_index(drop=True)
        i.drop(['����', 'ʱ��'], axis=1,inplace=True)
        for j in range(0,8):
            small_list.extend(list(i.iloc[j]))
        print('oooo',list(i.iloc[j]))
        the_big_list.append(small_list)
        print(i)
    test.groupby('����').apply(day_data)
    #��ȡ��ɣ����з���
    #print(train.head())
    print(the_big_list)
    the_column_list=[]
    for i in ['2','5','8','11','14','17','20','23']:
        for j in ['���ն�','����','����','�¶�','ʪ��','��ѹ']:
            the_column_list.append(j+i)
    the_test=pd.DataFrame(the_big_list,columns=the_column_list)
    #����һ��  ɾȥ��һ��
    the_test.drop([0],inplace=True)
    print(the_test)
    the_test.to_csv('../data/extract_test_data.csv')
    pass
if __name__=="__main__":
    print('�Գ�ʼ���ݰ���ʱ���ƽչ')
    ori_data_flat()