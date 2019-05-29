import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import missingno as msno
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xgboost as xgb
import seaborn as sns
import tensorflow as tf

warnings.filterwarnings("ignore")


train=pd.read_csv('train_set.csv',na_values=[-1, "unknown"])
test =pd.read_csv('test_set.csv',na_values=[-1, "unknown"])

#把列分为类别列，bin列和连续值列
bincol = ["default", "housing", "loan"]
catcol = ["marital", "education", "contact", "poutcome", "job"]
othercol = ['age', 'balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous']

def rename_col(st):
    if st in bincol:
        return st+"_bin"
    elif st in catcol:
        return st + "_cat"
    else:
        return st

train.columns = train.columns.map(lambda x:rename_col(x))
test.columns = test.columns.map(lambda x:rename_col(x))
bincol = list(map(lambda x: x+"_bin", bincol))
catcol = list(map(lambda x: x+"_cat", catcol))

misssing_col = train.isnull().any()[train.isnull().any()]

col_missing = train.isnull().any()[train.isnull().any()].index

miss_cat = [x for x in col_missing if x in catcol] + ["pdays"]
train[miss_cat] = train[miss_cat].replace(np.nan,-1)
#对于连续特征：

len_train = len(train)
len_test = len(test)

miss_nocat = [x for x in col_missing if x not in miss_cat]
train['pdays_miss'] = np.zeros(len_train)
train['pdays_miss'] [train.pdays.isnull()] = 1
test['pdays_miss'] = np.zeros(len_test)
test['pdays_miss'] [test.pdays.isnull()] = 1
train[miss_nocat] = train[miss_nocat].replace(np.nan,train[miss_nocat].median())
test[miss_nocat] = test[miss_nocat].replace(np.nan, train[miss_nocat].median())

poutcome_mapping = {-1:-1,"other":0,"success":1,"failure":-2}
train["poutcome_cat"] = train["poutcome_cat"].map(poutcome_mapping)
train["poutcome_pdays"] = train["poutcome_cat"] * train["pdays"]
test["poutcome_cat"] = test["poutcome_cat"].map(poutcome_mapping)
test["poutcome_pdays"] = test["poutcome_cat"] * test["pdays"]
for i in bincol:
    bin_mapping = {"yes":1, "no":0}
    train[i] = train[i].map(bin_mapping)
    test[i] = test[i].map(bin_mapping)

data = pd.concat([train,test])

feature=data.columns.tolist()

feature.remove('ID')
feature.remove('y')
sparse_feature= ['campaign','contact_cat','default_bin','education_cat','housing_bin','job_cat','loan_bin','marital_cat','month','poutcome_cat']
dense_feature=list(set(feature)-set(sparse_feature))
def get_new_columns(name,aggs):
    l=[]
    for k in aggs.keys():
        for agg in aggs[k]:
            if str(type(agg))=="<class 'function'>":
                l.append(name + '_' + k + '_' + 'other')
            else:
                l.append(name + '_' + k + '_' + agg)
    return l

for d in tqdm(sparse_feature):
    aggs={}
    for s in sparse_feature:
        aggs[s]=['count','nunique']
    for den in dense_feature:
        aggs[den]=['mean','max','min','std']
    aggs.pop(d)
    temp=data.groupby(d).agg(aggs).reset_index()
    temp.columns=[d]+get_new_columns(d,aggs)
    data=pd.merge(data,temp,on=d,how='left')

for s in catcol:
    data=pd.concat([data,pd.get_dummies(data[s],prefix=s+'_')],axis=1)
    data.drop(s,axis=1,inplace=True)

df_train=data[data['y'].notnull()]
df_test=data[data['y'].isnull()]

target=df_train['y']
df_train_columns=df_train.columns.tolist()
df_train_columns.remove('ID')
df_train_columns.remove('y')