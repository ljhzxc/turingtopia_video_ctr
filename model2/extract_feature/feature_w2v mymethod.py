from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy, kurtosis
import time
import gc
import random
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
sort_df = pd.read_pickle('../mydata/sort_df.pkl')
def emb(log, pivot, f, L=8):
    # word2vec算法
    # log为曝光日志，以pivot为主键，f为embedding的对象，flag为dev或test，L是embedding的维度
    print("w2v:", pivot, f)

    # 构造文档
    sentence = []
    dic = {}
    day = 0
    for item in log[['day', pivot, f]].values:
        if day != item[0]:
            for key in dic:
                sentence.append(dic[key])
            dic = {}
            day = item[0]
        try:
            dic[item[1]].append(str(int(item[2])))
        except:
            dic[item[1]] = [str(int(item[2]))]
    for key in dic:
        sentence.append(dic[key])
    print(len(sentence))
    # 训练Word2Vec模型
    print('training...')
    random.shuffle(sentence)
    model = Word2Vec(sentence, size=L, window=6, min_count=5,
                     sg=0, hs=1, seed=2019, workers=10)
    print('outputing...')
    # 保存文件
    values = set(log[f].values)
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model[str(v)])
            w2v.append(a)
        except:
            pass
    out_df = pd.DataFrame(w2v)
    names = [f]
    for i in range(L):
        names.append(pivot+'_w2v_embedding_'+f+'_'+str(L)+'_'+str(i))
    out_df.columns = names    
    return out_df
df = df.merge(emb(sort_df, 'deviceid', 'lng_lat'), on='lng_lat', how='left')
df = df.merge(emb(sort_df, 'lng_lat', 'newsid'), on='newsid', how='left')
df = df.merge(emb(sort_df, 'lng_lat', 'deviceid'), on='deviceid', how='left')
df = df.merge(emb(sort_df, 'deviceid', 'newsid'), on='newsid', how='left')
df = df.merge(emb(sort_df, 'newsid', 'deviceid'), on='deviceid', how='left')
df = df.merge(emb(sort_df, 'newsid', 'device_version'), on='device_version', how='left')
df = df.merge(emb(sort_df, 'newsid', 'lng_lat'), on='lng_lat', how='left')
del sort_df
gc.collect()
feature_list=df.columns.values.tolist()
feature_list=['id']+feature_list[41:]
df_w2v_32= df[feature_list]
df_w2v_32=reduce_mem(df_w2v_32)
df_w2v_32.to_pickle('../mydata/feature/feature_w2v_32_mymethod.pkl')