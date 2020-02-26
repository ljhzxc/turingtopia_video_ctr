import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy, kurtosis
import time
import gc
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
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
df = pd.read_pickle('../mydata/df_full/df_full_391_new.pkl')
labels = pd.read_pickle('../mydata/target.pkl')
print_time(start_time)
print('========================= tag feature ================================')
feature = pd.read_pickle('../mydata/feature/feature_tagsvd_20.pkl')
feature.drop(['deviceid'], axis=1, inplace=True)
df = pd.merge(df, feature, on=['id'], how='left')
print('========================= outertag feature ================================')
feature = pd.read_pickle('../mydata/feature/feature_outertagsvd_10.pkl')
df = pd.merge(df, feature, on=['id'], how='left')
print('========================= cross three feature ================================')
feature = pd.read_pickle('../mydata/feature/cross_feature2.pkl')
name=feature.columns.values.tolist()
df[name]=feature[name]
del feature
gc.collect()
print('=========================================== add useless feature ===========================================')
ignore_feature = ['id', 'ts']
# 观察得到的不重要特征
del_feats_3 = ['minute', 'lng', 'lat', 'lng_count', 'lat_count']

# # 之前一次训练得到的不重要特征
del_feats_2 = ["cross_['pos', 'lng_lat']_deviceid_ent",
               "cross_['pos', 'lng_lat']_newsid_ent",
               "cross_['newsid', 'pos']_deviceid_ent",
               'cross_lng_lat_newsid_nunique',
               "cross_['pos', 'lng_lat']_newsid_nunique",
               'cross_lng_lat_deviceid_ent',
               'cross_pos_lng_lat_count',
               "cross_['deviceid', 'lng_lat']_netmodel_nunique",
               'cross_lng_lat_newsid_ent',
               'cross_netmodel_pos_ent',
               "cross_['deviceid', 'pos']_netmodel_nunique",
               'lng_lat_count',
               "cross_['newsid', 'pos']_netmodel_nunique",
               'cross_pos_lng_lat_nunique',
               "cross_['pos', 'lng_lat']_netmodel_nunique",
               "cross_['pos', 'lng_lat']_deviceid_nunique",
               "cross_['netmodel', 'lng_lat']_deviceid_nunique",
               'cross_lng_lat_netmodel_nunique',
               'cross_netmodel_lng_lat_ent',
               'cross_lng_lat_deviceid_nunique']
# 队友40个不重要特征
del_feats_1 = ['pos_newsid_next3_exposure_ts_gap', 'cross_pos_lng_lat_short_nunique',
               'pos_deviceid_prev10_exposure_ts_gap', 'minute', 'pos_newsid_prev3_exposure_ts_gap',
               'cross_pos_netmodel_nunique', 'cross_deviceid_newsid_nunique', 'newsid_next10_exposure_ts_gap',
               'newsid_next3_exposure_ts_gap', 'pos_newsid_next10_exposure_ts_gap', 'lng_lat_short_count',
               'cross_deviceid_newsid_count_ratio', 'pos_deviceid_prev_day_click_count', 'lat',
               'cross_newsid_lng_lat_short_nunique', 'cross_newsid_netmodel_ent', 'cross_newsid_lng_lat_short_nunique_ratio_newsid_count',
               'cross_newsid_pos_count_ratio', 'cross_newsid_deviceid_nunique', 'cross_newsid_pos_nunique', 'device_vendor',
               'newsid_deviceid_emb_1', 'pos_newsid_next2_exposure_ts_gap', 'newsid_lng_lat_short_emb_6', 'cross_pos_newsid_nunique',
               'cross_newsid_deviceid_ent', 'lng_lat_short_prev3_exposure_ts_gap', 'pos_lng_lat_short_next10_exposure_ts_gap',
               'pos_deviceid_lng_lat_short_prev2_exposure_ts_gap', 'cross_deviceid_netmodel_nunique',
               'pos_newsid_next5_exposure_ts_gap', 'pos_deviceid_lng_lat_short_prev5_exposure_ts_gap',
               'pos_deviceid_lng_lat_short_prev10_exposure_ts_gap', 'deviceid_newsid_deviceid_deepwalk_embedding_16_14',
               'lng_lat', 'cross_netmodel_lng_lat_short_nunique_ratio_netmodel_count',
               'pos_deviceid_lng_lat_short_prev3_exposure_ts_gap', 'newsid_count',
               'lng', 'cross_newsid_netmodel_nunique']
del_feats = set(del_feats_1 +del_feats_2+ del_feats_3)
all_feature_list = df.columns.values.tolist()
for col in del_feats:
    if col in all_feature_list:
        ignore_feature.append(col)
print('=========================================== delete useless feature ===========================================')
df.drop(ignore_feature, axis=1, inplace=True)
gc.collect()
print_time(start_time)
print('=============================================== data partitioning ===============================================')
train_num = len(labels)
train_df = df[:train_num].reset_index(drop=True)
test_df = df[train_num:].reset_index(drop=True)
del df
gc.collect()

train_idx = train_df[train_df['day'] < 10].index.tolist()
val_idx = train_df[train_df['day'] == 10].index.tolist()

train_x = train_df.iloc[train_idx].reset_index(drop=True)
train_y = labels['target'][train_idx]
val_x = train_df.iloc[val_idx].reset_index(drop=True)
val_y = labels['target'][val_idx]
del train_x['day'], val_x['day'], train_df['day'], test_df['day']
gc.collect()
print_time(start_time)
print('=============================================== training validate ===============================================')
clf = CatBoostClassifier(
    iterations=20000,
    learning_rate=0.08,
    eval_metric='AUC',
    use_best_model=True,
    random_seed=475,
    task_type='GPU',
    devices='2',
    early_stopping_rounds=500,
    loss_function='Logloss',
    depth=7,
    verbose=200
#     save_snapshot=True,
#     #snapshot_file='',
#     snapshot_interval=300
)
print('************** training **************')
clf.fit(
    train_x, train_y,
    # cat_features=cate_cols,
    # sample_weight=local_weights,
    eval_set=[(val_x, val_y)],
)
print_time(start_time)
print('************** validate predict **************')
fea_imp_list = []
fea_imp_list.append(clf.feature_importances_)
best_rounds = clf.best_iteration_
best_auc = clf.best_score_['validation']['AUC']
val_pred = clf.predict_proba(val_x)[:, 1]
print_time(start_time)
print('=============================================== threshold search ===============================================')
# f1阈值敏感，对阈值做一个简单的迭代搜索
t0 = 0.05
v = 0.002
best_t = t0
best_f1 = 0
for step in range(160,185):
    curr_t = t0 + step * v
    y = [1 if x >= curr_t else 0 for x in val_pred]
    curr_f1 = f1_score(val_y, y)
    if curr_f1 > best_f1:
        best_t = curr_t
        best_f1 = curr_f1
        print('step: {}   best threshold: {}   best f1: {}'.format(
            step, best_t, best_f1))
print('search finish.')

val_pred = [1 if x >= best_t else 0 for x in val_pred]
print('\nbest auc:', best_auc)
print('best f1:', f1_score(val_y, val_pred))
print('validate mean:', np.mean(val_pred))
print_time(start_time)
print('=============================================== feat importances_1 ===============================================')
fea_imp_list = []
fea_imp_list.append(clf.feature_importances_)
print('=============================================== training predict ===============================================')
clf = CatBoostClassifier(
    iterations=best_rounds+1000,
    learning_rate=0.08,
    eval_metric='AUC',
    use_best_model=True,
    random_seed=475,
    task_type='GPU',
    devices='2',
    early_stopping_rounds=500,
    loss_function='Logloss',
    depth=7,
    verbose=200
    #save_snapshot=True,
    #snapshot_file='',
    #snapshot_interval=300
)
print('************** training **************')
clf.fit(
    train_df, labels['target'],
    # cat_features=cate_cols,
    # sample_weight=local_weights,
    eval_set=[(train_df, labels['target'])],
)
print_time(start_time)
print('************** test predict **************')
sub = pd.read_csv('../dataset/sample.csv')
sub['target'] = clf.predict_proba(test_df)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print_time(start_time)
print('=============================================== feat importances_mean ===============================================')
fea_imp_dict = dict(zip(train_df.columns.values,np.mean(fea_imp_list, axis=0)))
fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
for f, imp in fea_imp_item:
    print('{} = {}'.format(f, imp))
import datetime
print('========================================== save parameter==========================================')

THRESHOLD = str(best_t)
# 训练日期
DATE = datetime.datetime.now().strftime('%m-%d_%H:%M')
# 模型名称
MODEL = 'CatBoost'
# 迭代次数
BEST_ROUNDS = best_rounds
# 描述
DESCRIBE = '(LR0.08_long_twotag_svd)'
FEATURE_NUMBER = str(len(train_df.columns.values.tolist()))
path = '../sub/{}_{}_{}_{:.7f}_{:.7f}_{}_{}_{}'.format(
    DATE, MODEL, FEATURE_NUMBER, best_auc, best_f1, THRESHOLD, BEST_ROUNDS, DESCRIBE)
os.mkdir(path)
print('=============================================== model save ===============================================')
from sklearn.externals import joblib
joblib.dump(clf, path+'/model.pkl')
print('=============================================== sub save ===============================================')
feature_importances = pd.DataFrame(fea_imp_item, columns=['feature_name', 'importance'])
feature_importances.to_csv(path+'/FeaImportance_{}_{}_{}_{}_{}_{}_{}.csv'.format(DATE, FEATURE_NUMBER,
                                                                                 best_auc, best_f1, sub['target'].mean(), THRESHOLD, BEST_ROUNDS), index=False)
sub.to_csv(path+'/sub_prob_{}_{}_{}_{}_{}_{}_{}.csv'.format(DATE, FEATURE_NUMBER,
                                                            best_auc, best_f1, sub['target'].mean(), THRESHOLD, BEST_ROUNDS), index=False)
sub['target'] = sub['target'].apply(lambda x: 1 if x >= best_t else 0)
sub.to_csv(path+'/sub_{}_{}_{}_{}_{}_{}_{}.csv'.format(DATE, FEATURE_NUMBER,
                                                       best_auc, best_f1, sub['target'].mean(), THRESHOLD, BEST_ROUNDS), index=False)
print('finish.')
print_time(start_time)                                                                                