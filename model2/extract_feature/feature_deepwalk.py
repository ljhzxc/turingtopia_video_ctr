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
def deepwalk(df, f1, f2):
    L = 16
    #Deepwalk算法，
    print("deepwalk:",f1,f2)
    #构建图
    dic={}
    for item in df[[f1,f2]].values:
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_'+str(int(item[1]))].add('user_'+str(int(item[0])))
        except:
            dic['item_'+str(int(item[1]))]=set(['user_'+str(int(item[0]))])
        try:
            dic['user_'+str(int(item[0]))].add('item_'+str(int(item[1])))
        except:
            dic['user_'+str(int(item[0]))]=set(['item_'+str(int(item[1]))])
    dic_cont={}
    for key in dic:
        dic[key]=list(dic[key])
        dic_cont[key]=len(dic[key])
    print("creating")     
    #构建路径
    path_length=24
    sentences=[]
    length=[]
    for key in dic:
        sentence=[key]
        while len(sentence)!=path_length:
            key=dic[sentence[-1]][random.randint(0,dic_cont[sentence[-1]]-1)]
            if len(sentence)>=2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences)%100000==0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    #训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, size=L, window=4,min_count=1,sg=1, workers=10,iter=20)
    print('outputing...')
    #输出
    values=set(df[f1].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model['user_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df1=pd.DataFrame(w2v)
    names=[f1]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_embedding_'+str(L)+'_'+str(i))
    out_df1.columns = names
    print(out_df1.head())
    
    ########################
    values=set(df[f2].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model['item_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df2=pd.DataFrame(w2v)
    names=[f2]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_emb_'+str(L)+'_'+str(i))
    out_df2.columns = names
    print(out_df2.head())
    return (out_df1, out_df2)
emb_cols = [
    ['deviceid', 'newsid'],
    ['lng_lat', 'newsid']
]
for f1, f2 in emb_cols:
    out_df1, out_df2 = deepwalk(sort_df, f1, f2)
df = df.merge(out_df1, on='deviceid', how='left')
df = df.merge(out_df2, on='newsid', how='left')
del out_df1, out_df2
gc.collect()
feature_list=df.columns.values.tolist()
feature_list=['id']+feature_list[41:]
len(feature_list)
df_deepwalk_32= df[feature_list]
df_deepwalk_32.to_pickle('../mydata/feature/feature_deepwalk_32.pkl')