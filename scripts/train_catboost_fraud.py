#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install catboost')


# In[2]:


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os


# In[4]:


train = pd.read_csv("train.csv")


# In[5]:


def preprocess(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
    df["hour"] = df["transaction_time"].dt.hour
    df["dayofweek"] = df["transaction_time"].dt.dayofweek

    drop_cols = ["transaction_time", "name_1", "name_2", "street", "post_code"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    cat_features = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs"]
    num_features = ["amount","lat","lon","population_city","merchant_lat","merchant_lon","hour","dayofweek"]

    for c in cat_features:
        df[c] = df[c].astype(str).fillna("unknown")
    for c in num_features:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

    feat_cols = cat_features + num_features
    return df[feat_cols], cat_features, num_features

X_all, cat_features, num_features = preprocess(train.drop(columns=["target"]), is_train=True)
y_all = train["target"].astype(int)


# In[6]:


X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

model = CatBoostClassifier(
    cat_features=cat_features,
    random_seed=42,
    verbose=False
)
model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)


# In[7]:


os.makedirs("model", exist_ok=True)
model.save_model("model/model.cbm")

with open("model/features.txt", "w") as f:
    for c in X_all.columns:
        f.write(c + "\n")


# In[ ]:


model.cbm, features.txt

