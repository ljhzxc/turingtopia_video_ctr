import random
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy, kurtosis
import time
import gc
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('max_colwidth', 100)
pd.set_option('display.width', 1000)
start_time = time.time()
print('=============================================== tool definition ===============================================')


def print_time(start_time):
    print("run time: %dmin %ds, %s." % ((time.time() - start_time) /
                                        60, (time.time() - start_time) % 60, time.ctime()))


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


def kfold_static(train_df, test_df, f, label):
    print("K-fold static:", f+'_'+label)
    # K-fold positive and negative num
    # avg_rate=train_df[label].mean()
    num = len(train_df)//5
    index = [0 for i in range(num)]+[1 for i in range(num)]+[2 for i in range(num)]+[
        3 for i in range(num)]+[4 for i in range(len(train_df)-4*num)]
    random.shuffle(index)
    train_df['index'] = index
    # 五折统计
    dic = [{} for i in range(5)]
    # dic_all={}
    for item in train_df[['index', f, label]].values:
        try:
            dic[item[0]][item[1]].append(item[2])
        except:
            dic[item[0]][item[1]] = []
            dic[item[0]][item[1]].append(item[2])
    print("static done!")
    # 构造训练集的五折特征，均值，中位数等
    mean = []
    std = []
    cache = {}
    for item in train_df[['index', f]].values:
        if tuple(item) not in cache:
            temp = []
            for i in range(5):
                if i != item[0]:
                    try:
                        temp += dic[i][item[1]]
                    except:
                        pass
            if len(temp) == 0:
                cache[tuple(item)] = [-1]*5
            else:
                cache[tuple(item)] = [np.mean(temp), np.std(temp)]
        temp = cache[tuple(item)]
        mean.append(temp[0])
        std.append(temp[1])
    del cache
    train_df[f+'_'+label+'_mean'] = mean
    train_df[f+'_'+label+'_std'] = std
    print("train done!")

    # 构造测试集的五折特征，均值，中位数等
    mean = []
    std = []
    cache = {}
    for uid in test_df[f].values:
        if uid not in cache:
            temp = []
            for i in range(5):
                try:
                    temp += dic[i][uid]
                except:
                    pass
            if len(temp) == 0:
                cache[uid] = [-1]*5
            else:
                cache[uid] = [np.mean(temp), np.std(temp)]
        temp = cache[uid]
        mean.append(temp[0])
        std.append(temp[1])

    test_df[f+'_'+label+'_mean'] = mean
    test_df[f+'_'+label+'_std'] = std
    print("test done!")
    del train_df['index']
    print(f+'_'+label+'_mean')
    print(f+'_'+label+'_std')

    print('avg of mean', np.mean(
        train_df[f+'_'+label+'_mean']), np.mean(test_df[f+'_'+label+'_mean']))
    print('avg of std', np.mean(
        train_df[f+'_'+label+'_std']), np.mean(test_df[f+'_'+label+'_std']))


print('=============================================== read data ===============================================')
df = pd.read_pickle('../mydata/df.pkl')
labels = pd.read_pickle('../mydata/target.pkl')
train_df = df.loc[df['day'] < 11]
test_df = df.loc[df['day'] > 10]
train_df['label'] = labels
print('=============================================== Feature K-fold ===============================================')
kfold_static(train_df, test_df, 'guid', 'label')
kfold_static(train_df, test_df, 'deviceid', 'label')
kfold_static(train_df, test_df, 'newsid', 'label')
kfold_static(train_df, test_df, 'pos', 'label')
#########################################################
kfold_static(train_df, test_df, 'location_4', 'label')
kfold_static(train_df, test_df, 'device_version', 'label')
kfold_static(train_df, test_df, 'location_3', 'label')
kfold_static(train_df, test_df, 'lng_lat', 'label')

train_df.drop(['label'], axis=1, inplace=True)
df = pd.concat([train_df, test_df])
df.columns.values.tolist()
feature = df[['id', 'guid_label_mean',
              'guid_label_std',
              'deviceid_label_mean',
              'deviceid_label_std',
              'newsid_label_mean',
              'newsid_label_std',
              'pos_label_mean',
              'pos_label_std']]
feature = reduce_mem(feature)
feature.to_pickle('../mydata/feature/feature_kfold_8.pkl')
