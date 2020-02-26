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
from gensim.models import Word2Vec
import re
user = pd.read_csv('../dataset/user.csv')
countrow = 0
userlist = []
itemlist = []
ratelist = []
for item in user[user['tag']==user['tag']][['deviceid','tag']].values:
    ta = item[1]

    countrow +=1
    if countrow% 10000==0:print(countrow)

    try:
        tasp = [[re.split(':|_',tt)[0], float(re.split(':|_',tt)[2])] for tt in  ta.split('|') if len(tt.split(':'))>1]

        for tas in tasp:
            userlist.append(item[0])
            itemlist.append(tas[0])

            ratelist.append(tas[1])

    except:
        print(ta)
train = pd.DataFrame(zip(userlist, itemlist, ratelist), columns=['userID', 'itemsID', 'rating'])
train = train.sample(frac=1.0)
train = train.reset_index(drop=True)
train.describe(percentiles=[.01,.99])
train = train[train['rating']<1.610884e+01]
train = train[train['rating']>4.980915e-02]
import surprise
import time
train_set = surprise.Dataset.load_from_df(train, reader=surprise.Reader(rating_scale=(0.06, 9.96))).build_full_trainset()
svd = surprise.SVD(random_state=0, n_factors=20, n_epochs=300, verbose=True, lr_all=0.0005)
start_time = time.time()
svd.fit(train_set)
train_time = time.time() - start_time
user_id_uni = train['userID'].unique()
w2v=[]
for us, svdfea in zip(user_id_uni, svd.pu):
    a = [us]
    a.extend(list(svdfea))
    w2v.append(a)
out_df = pd.DataFrame(w2v, columns=['deviceid']+['tag_scd_'+str(i)+ '_feature' for i in range(20)])
train = pd.read_csv('../dataset/train.csv')
del train['target'], train['timestamp']
test_df = pd.read_csv('../dataset/test.csv')
df = pd.concat([train, test_df], axis=0, ignore_index=False)
df=df[['id','deviceid']]
df = df.merge(out_df, on='deviceid', how='left')
df.to_pickle('../mydata/feature/feature_tagsvd_20_new.pkl')

