import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from lightgbm.sklearn import LGBMClassifier
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
print('=============================================== read data ===============================================')
df = pd.read_pickle('../mydata/df.pkl')
df=df[['id','deviceid', 'newsid', 'pos', 'netmodel', 'lng_lat_short','deviceid_count','newsid_count','pos_count','netmodel_count','lng_lat_short_count']]
gc.collect()
print('=============================================== cross feature ===============================================')
# 二阶交叉特征
cross_cols = ['deviceid', 'newsid', 'pos', 'netmodel', 'lng_lat_short']
for f in cross_cols:
    for col in cross_cols:
        if col == f:
            continue
        print('------------------ {} {} ------------------'.format(f, col))
        df = df.merge(df[[f, col]].groupby(f, as_index=False)[col].agg({
            'cross_{}_{}_nunique'.format(f, col): 'nunique',
            'cross_{}_{}_ent'.format(f, col): lambda x: entropy(x.value_counts() / x.shape[0]) # 熵
        }), on=f, how='left')
        if 'cross_{}_{}_count'.format(f, col) not in df.columns.values and 'cross_{}_{}_count'.format(col, f) not in df.columns.values:
            df = df.merge(df[[f, col, 'id']].groupby([f, col], as_index=False)['id'].agg({
                'cross_{}_{}_count'.format(f, col): 'count' # 共现次数
            }), on=[f, col], how='left')
        if 'cross_{}_{}_count_ratio'.format(col, f) not in df.columns.values:
            df['cross_{}_{}_count_ratio'.format(col, f)] = df['cross_{}_{}_count'.format(f, col)] / df[f + '_count'] # 比例偏好
        if 'cross_{}_{}_count_ratio'.format(f, col) not in df.columns.values:
            df['cross_{}_{}_count_ratio'.format(f, col)] = df['cross_{}_{}_count'.format(f, col)] / df[col + '_count'] # 比例偏好
        df['cross_{}_{}_nunique_ratio_{}_count'.format(f, col, f)] = df['cross_{}_{}_nunique'.format(f, col)] / df[f + '_count']
        print_time(start_time)
df = reduce_mem(df)
feature_list=df.columns.values.tolist()
feature_list=['id']+feature_list[11:]
df_cross_90= df[feature_list]
df_cross_90.to_pickle('../mydata/feature/feature_cross_90_short.pkl')