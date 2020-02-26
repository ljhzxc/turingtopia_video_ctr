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
click_df = pd.read_pickle('../mydata/click_df.pkl')
print('*************************** history feature ***************************')
for f in [
    ['deviceid'],
    ['newsid'],
    ['deviceid', 'pos'],
    ['newsid', 'pos'],
    ['deviceid', 'netmodel'],
    ['deviceid', 'lng_lat_short'],
    ['pos', 'lng_lat_short'],
    ['netmodel', 'lng_lat_short'],
    ['deviceid', 'pos', 'netmodel'],
    ['deviceid', 'netmodel', 'lng_lat_short']
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    # 对前一天的点击次数进行统计
    tmp = click_df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)[
        'id'].agg({'_'.join(f) + '_prev_day_click_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_prev_day_click_count'] = df['_'.join(f) +
                                                   '_prev_day_click_count'].fillna(0)
    df.loc[df['day'] == 8, '_'.join(f) + '_prev_day_click_count'] = None

    # 对前一天的曝光量进行统计
    tmp = df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)['id'].agg({
        '_'.join(f) + '_prev_day_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_prev_day_count'] = df['_'.join(f) +
                                             '_prev_day_count'].fillna(0)
    df.loc[df['day'] == 8, '_'.join(f) + '_prev_day_count'] = None

    # 计算前一天的点击率
    df['_'.join(f) + '_prev_day_ctr'] = df['_'.join(f) + '_prev_day_click_count'] / (
        df['_'.join(f) + '_prev_day_count'] + df['_'.join(f) + '_prev_day_count'].mean())

    del tmp
    print_time(start_time)

print('*************************** today statistic feature ***************************')
for f in [
    ['deviceid'],
    ['newsid'],
    ['deviceid', 'pos'],
    ['newsid', 'pos'],
    ['deviceid', 'netmodel'],
    ['deviceid', 'lng_lat_short'],
    ['pos', 'lng_lat_short'],
    ['netmodel', 'lng_lat_short'],
    ['deviceid', 'pos', 'netmodel'],
    ['deviceid', 'netmodel', 'lng_lat_short']
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    # 对今天的曝光量进行统计
    tmp = df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)['id'].agg({
        '_'.join(f) + '_today_allday_count': 'count'})
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_today_allday_count'] = df['_'.join(f) +
                                                 '_today_allday_count'].fillna(0)
    # 对今天每个小时的曝光量进行统计
    tmp = df[f + ['day', 'id', 'hour']].groupby(f + ['day', 'hour'], as_index=False)['id'].agg({
        '_'.join(f) + '_today_everyhour_count': 'count'})
    df = df.merge(tmp, on=f + ['day', 'hour'], how='left')
    df['_'.join(f) + '_today_everyhour_count'] = df['_'.join(f) +
                                                    '_today_everyhour_count'].fillna(0)
    # 对今天每分钟的曝光量进行统计
    tmp = df[f + ['day', 'id', 'hour', 'minute']].groupby(f + ['day', 'hour', 'minute'], as_index=False)['id'].agg({
        '_'.join(f) + '_today_everyminute_count': 'count'})
    df = df.merge(tmp, on=f + ['day', 'hour', 'minute'], how='left')
    df['_'.join(f) + '_today_everyminute_count'] = df['_'.join(f) +
                                                      '_today_everyminute_count'].fillna(0)
    del tmp
    print_time(start_time)
print('*************************** future stats ***************************')
for f in [
    ['deviceid'],
    ['newsid'],
    ['deviceid', 'pos'],
    ['newsid', 'pos'],
    ['deviceid', 'netmodel'],
    ['deviceid', 'lng_lat_short'],
    ['pos', 'lng_lat_short'],
    ['netmodel', 'lng_lat_short'],
    ['deviceid', 'pos', 'netmodel'],
    ['deviceid', 'netmodel', 'lng_lat_short']
]:
    print('------------------ {} ------------------'.format('_'.join(f)))
    # 前1小时的曝光量
    tmp = df[f + ['total_hour', 'id']].groupby(f + ['total_hour'], as_index=False)[
        'id'].agg({'tmp_hour_count': 'count'})
    col_name = '_'.join(f) + f'_pre1_hour_count'
    tmp['total_hour'] += 1
    df = df.merge(tmp.rename(
        columns={'tmp_hour_count': col_name}), on=f + ['total_hour'], how='left')
    del tmp
    # 前1分钟的曝光量
    tmp = df[f + ['total_minute', 'id']].groupby(f + ['total_minute'], as_index=False)[
        'id'].agg({'tmp_minute_count': 'count'})
    col_name = '_'.join(f) + f'_pre1_minute_count'
    tmp['total_minute'] += 1
    df = df.merge(tmp.rename(
        columns={'tmp_minute_count': col_name}), on=f + ['total_minute'], how='left')
    del tmp
    # 后1小时的曝光量
    tmp = df[f + ['total_hour', 'id']].groupby(f + ['total_hour'], as_index=False)[
        'id'].agg({'tmp_hour_count': 'count'})
    col_name = '_'.join(f) + f'_late1_hour_count'
    tmp['total_hour'] -= 1
    df = df.merge(tmp.rename(
        columns={'tmp_hour_count': col_name}), on=f + ['total_hour'], how='left')
    del tmp
    # 后1分钟的曝光量
    tmp = df[f + ['total_minute', 'id']].groupby(f + ['total_minute'], as_index=False)[
        'id'].agg({'tmp_minute_count': 'count'})
    col_name = '_'.join(f) + f'_late1_minute_count'
    tmp['total_minute'] -= 1
    df = df.merge(tmp.rename(
        columns={'tmp_minute_count': col_name}), on=f + ['total_minute'], how='left')
    del tmp
    print_time(start_time)
del click_df
df = reduce_mem(df)
feature_list=df.columns.values.tolist()
df_history_44 = df[['id']+feature_list[41:]]
def missing_data(df):
    total=df.isnull().sum().sort_values(ascending=False)
    percent=(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data=pd.concat([total,percent],axis=1,keys=['Total_missing','Percent'])
    return missing_data
tmp=missing_data(df_history_44)
df_history_44.drop(['newsid_pos_late1_minute_count','newsid_pos_pre1_minute_count','newsid_late1_minute_count','newsid_pre1_minute_count'], axis=1, inplace=True)
df_history_44.to_pickle('../mydata/feature/feature_history_new_96.pkl')