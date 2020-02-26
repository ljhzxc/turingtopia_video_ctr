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
    print("run time: %dmin %ds, %s." % ((time.time() - start_time)/60, (time.time() - start_time)%60, time.ctime()))

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
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df
print('=============================================== read data ===============================================')
df=pd.read_pickle('../mydata/df.pkl')
sort_df=pd.read_pickle('../mydata/sort_df.pkl')
print('*************************** exposure_ts_gap ***************************')
for f in [
    # 一阶
    ['deviceid'], ['newsid'], ['lng_lat_short']
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    tmp = sort_df[f + ['ts']].groupby(f)
    # 前x次、后x次曝光到当前的时间差
    for gap in [1, 2, 3, 4, 5, 6, 15]:
        sort_df['{}_prev{}_exposure_ts_gap'.format(
            '_'.join(f), gap)] = tmp['ts'].shift(0) - tmp['ts'].shift(gap)
        sort_df['{}_next{}_exposure_ts_gap'.format(
            '_'.join(f), gap)] = tmp['ts'].shift(-gap) - tmp['ts'].shift(0)
        tmp2 = sort_df[
            f + ['ts', '{}_prev{}_exposure_ts_gap'.format(
                '_'.join(f), gap), '{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
        ].drop_duplicates(f + ['ts']).reset_index(drop=True)
        df = df.merge(tmp2, on=f + ['ts'], how='left')
        del sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del tmp2
    del tmp
    df = reduce_mem(df)
    print_time(start_time)
print('*************************** exposure_ts_gap ***************************')
for f in [
    # 二阶
    ['pos', 'deviceid'], ['pos', 'newsid'], ['pos', 'lng_lat_short'], ['netmodel', 'deviceid'],
    ['netmodel', 'lng_lat_short'], ['deviceid', 'lng_lat_short'], ['netmodel', 'location_4'],
    # 三阶
    ['pos', 'netmodel', 'deviceid'], ['pos', 'deviceid', 'lng_lat_short'],
    ['netmodel', 'deviceid', 'lng_lat_short'], ['pos', 'netmodel', 'lng_lat_short'],
    # 四阶
    ['pos', 'netmodel', 'deviceid', 'lng_lat_short']
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    tmp = sort_df[f + ['ts']].groupby(f)
    # 前x次、后x次曝光到当前的时间差
    for gap in [1, 2, 3, 4, 5, 6, 15]:
        sort_df['{}_prev{}_exposure_ts_gap'.format(
            '_'.join(f), gap)] = tmp['ts'].shift(0) - tmp['ts'].shift(gap)
        sort_df['{}_next{}_exposure_ts_gap'.format(
            '_'.join(f), gap)] = tmp['ts'].shift(-gap) - tmp['ts'].shift(0)
        tmp2 = sort_df[
            f + ['ts', '{}_prev{}_exposure_ts_gap'.format(
                '_'.join(f), gap), '{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
        ].drop_duplicates(f + ['ts']).reset_index(drop=True)
        df = df.merge(tmp2, on=f + ['ts'], how='left')
        del sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del tmp2
    del tmp
    df = reduce_mem(df)
    print_time(start_time)
del sort_df
gc.collect()
feature_list=df.columns.values.tolist()
feature_list=['id']+feature_list[41:]
len(feature_list)
df_timegap_196= df[feature_list]
df_timegap_196.to_pickle('../mydata/feature/feature_timegap_new_210.pkl')