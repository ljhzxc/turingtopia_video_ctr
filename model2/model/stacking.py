import pandas as pd
import math
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('max_colwidth', 100)
pd.set_option('display.width', 1000)
sub1 = pd.read_csv('wu_model1.csv')
sub2 = pd.read_csv('wu_model2.csv')
sub1['target_2'] = sub2['target']
sub1['target'] = sub1['target']*0.5 + sub1['target_2']*0.5
del sub1['target_2']
sub2=pd.read_csv('../02-06_02:22_CatBoost_464_0.9823908_0.8115741_0.376_16034_(LR0.08_long_tag_svd)-0.8233/sub_prob_02-06_02:22_464_0.9823907613754272_0.8115740850408621_0.10292458905626266_0.376_16034.csv')
sub3=pd.read_csv('mymodel_0.82322_cross_4_lda.csv')
sub2['target_2']=sub3['target']
sub2['target']= sub2['target']*0.5 + sub2['target_2']*0.5
del sub2['target_2']
sub1['target_2']=sub2['target']
sub3=pd.read_csv('82.067.csv')
sub1['target_3']=sub3['target']
sub1['target'] = sub1['target']*0.5 + sub1['target_2']*0.35 + sub1['target_3']*0.15
del sub1['target_2'],sub1['target_3']
threshold = sorted(sub1['target'], reverse=True)[int(len(sub1['target'])*0.11)]
sub1['target'] = sub1['target'].apply(lambda x: 1 if x >= threshold else 0)
sub1.to_csv('sub_stacking_0206_wu_two(0.5)_mynewtwo(0.35)_0.82067(0.15).csv', index=False)
