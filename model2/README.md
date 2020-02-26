# 代码说明

## 1. 环境配置和所需依赖库

- 操作系统：Linux、Windows、MacOS，64G以上内存
- pandas
- numpy

- scikit-learn

- scipy

- gensim

- catboost
- geohash
- surprise

## 2. 代码按照以下顺序运行

data_process/data_process.py：数据清洗，生成训练集

extract_feature/：特征构造，八个文件八组特征可并行运行

data_process/merge_feature.py：特征合并

model/train_catboost.py：模型训练

model/feature_select.py：特征选择

model/stacking.py：模型融合