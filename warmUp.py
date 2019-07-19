#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1:  Set up imports, define variables and function, clean data:
# Standard Imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings

# Turn Off Warnings:
filterwarnings('ignore')

# Sklearn Imports:
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, VarianceThreshold, SelectFromModel, SelectKBest
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV

# Obtain and Clean Data:
def read_clean():
    global X, y, test
    train_values_file = input('type path to train values file')
    train_labels_file = input('type path to train labels file')
    test_values_file = input('type path to test values file')
    X = pd.read_csv(train_values_file)
    y = pd.read_csv(train_labels_file)
    test_original = pd.read_csv(test_values_file)

    X.drop('patient_id', axis=1, inplace=True)
    y.drop('patient_id', axis=1, inplace=True)
    X['thal'] = X.thal.astype('category')
    X = pd.get_dummies(X, drop_first=True)
    test = test_original.drop('patient_id', axis=1)
    test['thal'] = test.thal.astype('category')
    test = pd.get_dummies(X, drop_first=True)

# 2)  Feature Selection / Feature Engineering
### For this warm-up, the data was so clean and simple I did not do any real Feature
# Engineering.

# For loop to explore best features using kBest(chi2 algorithm):
def kBest_features():
    global kb_feats
    cols = X.columns.values
    kb_feats = [] 

    for i in range(1, len(X.columns)+1):#len(X.columns)):
    
        skb = SelectKBest(score_func=chi2, k=i).fit(X,y)
        mask = skb.get_support()
    
        feats = []
    
        for bool, col in zip(mask, cols):
            if bool:
                feats.append(col)
            
        kb_feats.append(feats)

# 3) Model selection and tuning.  Model is hard coded into the function so will need to be 
# changed as we test new models.  Could automate to run multiple models at once but I'm 
# too lazy...

# Test models with best features list 
def best_features_for_algorithm():
    global sample
    best=[]
    i=0

    for kb in kb_feats:
        print(kb)
        X_train, X_test, y_train, y_test = train_test_split(X[kb], y, test_size=0.33, random_state=42)
        params = {'randomforestclassifier__n_estimators': [50,100,150]}
        pipe = Pipeline(steps=[('randomforestclassifier', RandomForestClassifier(verbose=1, n_estimators=50,
                                                          min_samples_leaf=2, 
                                                          criterion='gini',
                                                         n_jobs=-1))])

        gs = GridSearchCV(pipe, param_grid=params, cv=2)
        gs.fit(X_train, y_train)    
        pred = gs.predict_proba(X_test)
        sample = gs.predict_proba(test[kb])

#sample = model.predict(test[feats])
        print(i)
        print(gs.score(X_train, y_train))
        print(gs.score(X_test, y_test))
        print(log_loss(y_test, pred))
        print('\n')
        i+=1

# 4)  If model tuned and working well, submit the model for scoring.  
# Submit for scoring:
def set_up_submission():
    submit = pd.DataFrame(sample[90:], index=test_original.patient_id, columns=['heart_disease_present'])
    submit['heart_disease_present'] = submit.heart_disease_present.astype('float64')
    submit.to_csv('SVR.csv')   


# In[ ]:




