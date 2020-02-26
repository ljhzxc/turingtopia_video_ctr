import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
df.drop(['total_hour','total_minute'], axis=1, inplace=True)
feature= pd.read_pickle('../mydata/feature/feature_deepwalk_32.pkl')
df=pd.merge(df,feature,on=['id'], how='left')
del feature
gc.collect()
feature= pd.read_pickle('../mydata/feature/feature_w2v_32_mymethod.pkl')
df=pd.merge(df,feature,on=['id'], how='left')
del feature
gc.collect()
feature= pd.read_pickle('../mydata/feature/feature_cross_90.pkl')
df=pd.merge(df,feature,on=['id'], how='left')
del feature
gc.collect()
feature= pd.read_pickle('../mydata/feature/feature_history_new_96.pkl')
df=pd.merge(df,feature,on=['id'], how='left')
del feature
gc.collect()
feature= pd.read_pickle('../mydata/feature/feature_timegap_196.pkl')
df=pd.merge(df,feature,on=['id'], how='left')
del feature
gc.collect()
df.to_pickle('../mydata/df_full/df_full_391_new.pkl')