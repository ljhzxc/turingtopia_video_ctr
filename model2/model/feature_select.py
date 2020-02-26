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
print('=============================================== read data ===============================================')
df = pd.read_pickle('../mydata/df_full/df_full_499.pkl')
labels = pd.read_pickle('../mydata/target.pkl')
print_time(start_time)
ignore_feature = ['id', 'ts','minute','lng','lat','lng_count','lat_count']
df.drop(ignore_feature, axis=1, inplace=True)
gc.collect()
print_time(start_time)
print('=============================================== data partitioning ===============================================')
train_num = len(labels)
train_df = df[:train_num].reset_index(drop=True)
del df
gc.collect()

train_idx = train_df[train_df['day'] < 10].index.tolist()
train_x = train_df.iloc[train_idx].reset_index(drop=True)
train_y = labels['target'][train_idx]

del train_x['day'], train_df['day']
gc.collect()
print_time(start_time)
np.random.seed(475)
train_features = train_x.columns.values.tolist()
def get_feature_importances(train_x,train_y,shuffle,seed=None):    
    # Shuffle target if required
    y = train_y.copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = train_y.copy().sample(frac=1.0)
    
    clf = CatBoostClassifier(
        iterations=14000,
        learning_rate=0.08,
        eval_metric='AUC',
        random_seed=475,
        task_type='GPU',
        devices='2',
        early_stopping_rounds=200,
        loss_function='Logloss',
        depth=7,
        verbose=2000
    )

    clf.fit(
        train_x, y
    )

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = train_features
    imp_df["importance"] = clf.feature_importances_
    imp_df['train_score'] = clf.best_score_['learn']['AUC']
    
    return imp_df
# Get the actual importance, i.e. without shuffling
print('========================================== calculate actual_imp_df ==========================================')
actual_imp_df = get_feature_importances(train_x, train_y,shuffle=False)
actual_imp_df.to_csv('actual_imp_df.csv')
print_time(start_time)
null_imp_df = pd.DataFrame()
nb_runs = 100
import time
start = time.time()
print('========================================== calculate null_imp_df ==========================================')
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(train_x, train_y,shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp)
    null_imp_df.to_csv('null_imp_df.csv')
print_time(start_time)
print('======================================== calculate feature_scores ========================================')
feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps_importance = null_imp_df.loc[null_imp_df['feature']
                                             == _f, 'importance'].values
    f_act_imps_importance = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance'].mean(
    )
    gain_score = np.log(1e-10 + f_act_imps_importance / (1 +
                                                         np.percentile(f_null_imps_importance, 75)))  # Avoid didvide by zero
    feature_scores.append((_f, gain_score))
scores_df = pd.DataFrame(feature_scores, columns=[
                         'feature', 'importance_score'])
scores_df.sort_values("importance_score", inplace=True, ascending=False)
scores_df=scores_df.reset_index(drop=True)
scores_df.to_csv('importance.csv')
print_time(start_time)
print('finish.')