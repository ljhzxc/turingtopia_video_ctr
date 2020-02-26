import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
#from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy
from gensim.models import Word2Vec
import time
import gc
# 地理位置编码库
import geohash as gh
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
print('=============================================== read train ===============================================')
train_df = pd.read_csv('../dataset/train.csv')
labels = train_df['target'].values
target = pd.DataFrame(labels, columns=['target']) 
target.to_pickle('../mydata/target.pkl')
train_df['date'] = pd.to_datetime(
    train_df['ts'].apply(lambda x: time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
train_df['day'] = train_df['date'].dt.day
train_df['hour'] = train_df['date'].dt.hour
train_df['minute'] = train_df['date'].dt.minute
# 7号数据处理
train_df.loc[train_df['day'] == 7, 'hour'] = 0
train_df.loc[train_df['day'] == 7, 'minute'] = 0
train_df.loc[train_df['day'] == 7, 'day'] = 8
# train_num = train_df.shape[0]
print_time(start_time)
print('=============================================== click data ===============================================')
click_df = train_df[train_df['target'] == 1].sort_values(
    'timestamp').reset_index(drop=True)
click_df['exposure_click_gap'] = click_df['timestamp'] - click_df['ts']
click_df = click_df[click_df['exposure_click_gap'] >= 0].reset_index(drop=True)
click_df['date'] = pd.to_datetime(
    click_df['timestamp'].apply(lambda x: time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
click_df['day'] = click_df['date'].dt.day
click_df['hour'] = click_df['date'].dt.hour
click_df['minute'] = click_df['date'].dt.minute
click_df.loc[click_df['day'] == 7, 'day'] = 8
del train_df['target'], train_df['timestamp']
for f in ['date', 'target',]:
    del click_df[f]
# for f in ['date', 'exposure_click_gap', 'timestamp', 'ts', 'target', 'hour', 'minute']:
#     del click_df[f]
print_time(start_time)
print('=============================================== read test ===============================================')
test_df = pd.read_csv('../dataset/test.csv')
test_df['date'] = pd.to_datetime(
    test_df['ts'].apply(lambda x: time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
test_df['day'] = test_df['date'].dt.day
test_df['hour'] = test_df['date'].dt.hour
test_df['minute'] = test_df['date'].dt.minute
# 10号数据处理
test_df.loc[test_df['day'] == 10, 'hour'] = 0
test_df.loc[test_df['day'] == 10, 'minute'] = 0
test_df.loc[test_df['day'] == 10, 'day'] = 11
df = pd.concat([train_df, test_df], axis=0, ignore_index=False)
del train_df, test_df, df['date'], df['guid']
gc.collect()
print_time(start_time)
print('=============================================== location enc ===============================================')
# 根据经纬度编码位置，大致可认为是国家、省、市、区
def location_process(df):
    print('经纬度保留几位小数')
    df['lng_short'] = df['lng'].astype('str')
    df['lat_short'] = df['lat'].astype('str')
    df['lng_short'] = df['lng_short'].map(lambda x: x[0:7])
    df['lat_short'] = df['lat_short'].map(lambda x: x[0:6])
    print('经纬度连接确定具体位置')
    df['lng_lat_short'] = df['lng_short'] + '_' + df['lat_short']
    df['lng_lat'] = df['lng'].astype('str') + '_' + df['lat'].astype('str')
    print('编码位置')
    df['location_3'] = df.apply(
        lambda x: gh.encode(x['lat'], x['lng'], 3), axis=1)
    df['location_4'] = df.apply(
        lambda x: gh.encode(x['lat'], x['lng'], 4), axis=1)
    df['location_5'] = df.apply(
        lambda x: gh.encode(x['lat'], x['lng'], 5), axis=1)
    return df

df = location_process(df)
click_df = location_process(click_df)
click_df.to_pickle('../mydata/raw_data/click_df.pkl')
df.to_pickle('../mydata/raw_data/df.pkl')
print('=============================================== cate enc ===============================================')
sort_df = df.sort_values('ts').reset_index(drop=True)
cate_cols = [
    'deviceid', 'newsid', 'pos', 'app_version', 'device_vendor',
    'netmodel', 'osversion', 'lng', 'lat', 'device_version', 'lng_short',
    'lat_short', 'lng_lat_short', 'lng_lat', 'location_3', 'location_4', 'location_5'
]
for f in cate_cols:
    print(f)
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    click_df[f] = click_df[f].map(map_dict).fillna(-1).astype('int32')
    sort_df[f] = sort_df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = reduce_mem(df)
click_df = reduce_mem(click_df)
sort_df = reduce_mem(sort_df)
print_time(start_time)
# 计算总小时数和总分钟数
df['day'] = df['day'].astype(np.int64)
df['hour'] = df['hour'].astype(np.int64)
df['minute'] = df['minute'].astype(np.int64)
df['total_hour']=df['day'] * 24 + df['hour']
df['total_minute'] = df['total_hour']*60+df['minute']
df = reduce_mem(df)
df.to_pickle('../mydata/df.pkl')
sort_df.to_pickle('../mydata/sort_df.pkl')
click_df.to_pickle('../mydata/click_df.pkl')
