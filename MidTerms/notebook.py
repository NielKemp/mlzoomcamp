#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb


# In[2]:


#read in data
data = pd.read_csv('data.csv')
del data['Unnamed: 0']

data.head()


# In[14]:


data.describe()


# ## fix columns names

# In[3]:


data.rename(columns={'SeriousDlqin2yrs':'delinquent','RevolvingUtilizationOfUnsecuredLines':'revolveUtilUnsecured','NumberOfTime30-59DaysPastDueNotWorse':'times30to59dayslate','DebtRatio':'debtRatio', 'MonthlyIncome':'monthlyIncome', 'NumberOfOpenCreditLinesAndLoans':'totalCreditLines', 'NumberOfTimes90DaysLate':'times90DaysLate', 'NumberRealEstateLoansOrLines':'realEstateLines', 'NumberOfTime60-89DaysPastDueNotWorse':'times60to89dayslate', 'NumberOfDependents':'dependents'   }, inplace = True)


# ## data types

# In[93]:


data.dtypes


# # Exploratory Data Analysis
# ## target distribution

# In[94]:


data.delinquent.hist()


# ## missings

# In[68]:


print(data.isnull().sum())


# ## feature distribution

# ### revolveUtilUnsecured

# In[29]:


sns.distplot(data.revolveUtilUnsecured)


# In[84]:


sns.distplot(data.revolveUtilUnsecured[data.revolveUtilUnsecured  <= 1])


# In[32]:


sns.distplot(np.log1p(data.revolveUtilUnsecured))


# In[83]:


sns.distplot(np.log1p(data.revolveUtilUnsecured[data.revolveUtilUnsecured  < 1]))


# In[85]:


#data.count()
data.revolveUtilUnsecured[data.revolveUtilUnsecured < 1].count()


# In[97]:


print(data.revolveUtilUnsecured[data.revolveUtilUnsecured > 1].count())
print(data.revolveUtilUnsecured[(data.revolveUtilUnsecured > 1) & (data.delinquent  == 1)].count())


# In[98]:


print(data.revolveUtilUnsecured[data.revolveUtilUnsecured < 1].count())
print(data.revolveUtilUnsecured[(data.revolveUtilUnsecured < 1) & (data.delinquent  == 1)].count())


# ### age

# In[99]:


sns.distplot(data.age)


# In[105]:


print(data.revolveUtilUnsecured[data.age < 18].count())


# In[112]:


print(data.revolveUtilUnsecured[data.age > 70].count())
print(data.revolveUtilUnsecured[(data.age > 70) & (data.delinquent  == 1)].count())

print(data.revolveUtilUnsecured[data.age > 80].count())
print(data.revolveUtilUnsecured[(data.age > 80) & (data.delinquent  == 1)].count())

print(data.revolveUtilUnsecured[data.age > 90].count())
print(data.revolveUtilUnsecured[(data.age > 90) & (data.delinquent  == 1)].count())

print(data.revolveUtilUnsecured[data.age > 100].count())
print(data.revolveUtilUnsecured[(data.age > 100) & (data.delinquent  == 1)].count())


# ### times30to59dayslate        

# In[44]:


sns.distplot(data.times30to59dayslate)


# In[45]:


sns.distplot(data.times30to59dayslate[data.times30to59dayslate < 10])


# In[46]:


sns.distplot(np.log1p(data.times30to59dayslate))


# ### debtRatio               

# In[69]:


sns.distplot(data.debtRatio)


# In[75]:


sns.distplot(data.debtRatio[data.debtRatio < 1])


# In[113]:


sns.distplot(np.log1p(data.debtRatio))


# In[136]:


print(data.debtRatio[data.debtRatio > 1].count())
print(data.debtRatio[(data.debtRatio > 1) & (data.delinquent  == 1)].count())

print(data.debtRatio[(data.debtRatio > 1) & (data.monthlyIncome.notnull())].count())
print(data.debtRatio[(data.debtRatio > 1) & (data.delinquent  == 1) & (data.monthlyIncome.notnull())].count() )

print(data.debtRatio[data.debtRatio <= 1].count())
print(data.debtRatio[(data.debtRatio <= 1) & (data.delinquent  == 1)].count())


# In[116]:


print(2291/35137)
print(7735/114863)


# ### monthlyIncome    

# In[137]:


sns.distplot(data.monthlyIncome)


# In[138]:


sns.distplot(data.monthlyIncome[data.monthlyIncome.notnull()])


# In[53]:


sns.distplot(np.log1p(data.monthlyIncome))


# ### totalCreditLines 

# In[54]:


sns.distplot(data.totalCreditLines)


# In[55]:


sns.distplot(np.log1p(data.totalCreditLines))


# ### times90DaysLate 

# In[58]:


sns.distplot(data.times90DaysLate)


# In[60]:


sns.distplot(np.log1p(data.times90DaysLate))


# ### realEstateLines 

# In[61]:


sns.distplot(data.realEstateLines)


# In[62]:


sns.distplot(np.log1p(data.realEstateLines))


# ### times60to89dayslate       

# In[63]:


sns.distplot(data.times60to89dayslate)


# In[64]:


sns.distplot(np.log1p(data.times60to89dayslate))


# ### dependents

# sns.distplot(data.dependents)

# In[67]:


sns.distplot(np.log1p(data.dependents))


# # Summary of EDA and actions per variable
# 
# target (delinquent) : target has a very low postive rate, can't use accuracy as performance measure, will use ROC AUC
# 
# revolveUtilUnsecured : A few outliers outside of the expected range of 0 - 1, these have a higher rate of delinquency than those that are within the range. Apply log transform to bring tail in.
# 
# age : one individual younger than 18, not expected. will force this value to be equal to 18 atleast.             
# 
# times30to59dayslate: very long tail, log-transformation helps this out a bit, but not a lot
# 
# debtRatio: strange behaviour, expected range is 0-1, 20% of values lie above this, and only some can be explained by missing monthlyIncome values. Need to try models with and without this.
# 
# monthlyIncome : many missing values, Consider imputing with avg's, and creating indicator that corresponds with missing incomes.   
# 
# totalCreditLines : well populated, looks good with log transformation
# 
# times90DaysLate : apply log transform, deals with tail values a bit better
# 
# realEstateLines :  apply log transform, deals with tail values a bit better
# 
# times60to89dayslate : apply log transform, deals with tail values a bit better
#  
# dependents : a few missings, impute with 0's
# 
# One-hot-encoding isn't required, since there are no categorical values.

# # Apply data cleaning/transformation techniques

# In[4]:


data.revolveUtilUnsecured = np.log1p(data.revolveUtilUnsecured)
data.age[data.age < 18] = 18
data.times30to59dayslate = np.log1p(data.times30to59dayslate)

data.loc[data.monthlyIncome.isnull(), 'incomeNullInd'] = 1
data.incomeNullInd = data.incomeNullInd.fillna(0).astype(int)

incomeAvg = data.monthlyIncome.mean()

data.monthlyIncome.fillna(data.monthlyIncome.mean(), inplace = True)

data.totalCreditLines = np.log1p(data.totalCreditLines)
data.times90DaysLate = np.log1p(data.times90DaysLate)
data.realEstateLines = np.log1p(data.realEstateLines)
data.times60to89dayslate = np.log1p(data.times60to89dayslate)
data.dependents.fillna(0,inplace = True)


# # Train / Validation split
# 
# Train: 60%
# Validation: 20%
# Test: 20%

# In[5]:


df_full_train, df_test = train_test_split(data, test_size=0.20, random_state=42)
df_train, df_valid = train_test_split(df_full_train, test_size = 0.25, random_state = 42)

x_train = df_train
x_valid = df_valid
x_test = df_test
x_full_train = df_full_train

y_train = x_train.delinquent.values
y_valid = x_valid.delinquent.values
y_test = x_test.delinquent.values
y_full_train = x_full_train.delinquent.values

del x_train['delinquent']
del x_valid['delinquent']
del x_test['delinquent']
del x_full_train['delinquent']


# # Model Training
# 
# Feature importance:
#     - mutual Information
#     - ROC AUC as feature importance
# 
# Models to try: 
#     - logistic regression
#     - decision tree
#     - random forest
#     - XGBoost

# ### Mutual information

# In[12]:


features = x_train.columns
for f in features:
    mis = mutual_info_score(y_train, x_train[f])
    print(f'feature: {f}, \t\t\t mis: {mis}')
    


# ### ROC AUC Score

# In[14]:


features = x_train.columns
for f in features:
    auc = roc_auc_score(y_train, x_train[f])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -x_train[f])
    print(f'feature: {f}, \t\t\t mis: {auc}')


# ## Logistic Regression
# 
# Try features in this order:
#     - revolveUtilUnsecured
#     - monthlyincome
#     - age
#     - times30to59dayslate
#     - times60to89dayslate
#     - times90DaysLate
#     - debtRatio
#     - dependents
#     - monthlyIncome
#     - totalCreditLines
#     - incomeNullInd
#     

# In[6]:


#features = ['revolveUtilUnsecured','monthlyIncome','age','times30to59dayslate','times60to89dayslate','times90DaysLate','debtRatio','dependents','monthlyIncome','totalCreditLines','incomeNullInd']
features = ['revolveUtilUnsecured','times30to59dayslate','times60to89dayslate','times90DaysLate','incomeNullInd','debtRatio','dependents','monthlyIncome','totalCreditLines']
list = []

for f in features:
    list.append(f)
    
    model = LogisticRegression()
    model.fit(x_train[list], y_train)
      
    preds = model.predict_proba(x_valid[list])[:,1]
    
    auc = roc_auc_score(y_valid, preds)
    print(list)
    print(auc)
    print('****************************************************')
    
finalLogisticReg = LogisticRegression()
finalLogisticReg.fit(x_full_train[['revolveUtilUnsecured', 'times30to59dayslate', 'times60to89dayslate', 'times90DaysLate']], y_full_train)


# ## Decision Tree
# 
# Parameters to tune: 
#     - Max depth
#     - Min Samples Leaf

# In[72]:


scores = []
for depth_ in [2,3,4,5,6,7,8,9,10]:
    for sample_ in [10,50,100,250,500,1000,1500,3000,4500]:
        model =  DecisionTreeClassifier(max_depth = depth_, min_samples_leaf = sample_, random_state = 42)
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_valid)[:,1]
        auc = roc_auc_score(y_valid, preds)
        scores.append((depth_, sample_, auc))


# In[73]:


columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns = columns)

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns='max_depth', values=['auc'])
print(df_scores_pivot.round(3))
sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")


# In[7]:


finalDeicisionTree = DecisionTreeClassifier(max_depth = 8, min_samples_leaf = 500, random_state = 42)
finalDeicisionTree.fit(x_full_train, y_full_train)


# ## Random Forest
# 
# Parameters to tune: 
#     - n_estimators
#     - Max depth
#     - Min Samples Leaf

# In[ ]:


forestscores = []
for estimators_ in [10,20,50,100,200,500]:
    for depth_ in [2,3,4,5,6,7,8,9,10]:
        for samples_ in [10,50,100,250,500,1000,1500]:
            model =  RandomForestClassifier(n_estimators = estimators_, max_depth = depth_, min_samples_leaf = samples_, random_state = 42)
            model.fit(x_train, y_train)
            preds = model.predict_proba(x_valid)[:,1]
            auc = roc_auc_score(y_valid, preds)
            forestscores.append((estimators, depth_, samples_, auc))
            


# In[115]:


forestscores = []
counter = 1
for depth_ in [4,5,6,7,8,9,10]:
    for samples_ in [10,50,100,250,500,1000,1500]:
        model =  RandomForestClassifier(n_estimators = 10, max_depth = depth_, min_samples_leaf = samples_, random_state = 42)
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_valid)[:,1]
        auc = roc_auc_score(y_valid, preds)
        forestscores.append((depth_, samples_, auc))
        print(f'*************** {counter} forests completed ***************')
        counter = counter +1
            
 


# In[116]:


columns = ['max_depth', 'min_samples_leaf', 'auc']
df_forestscores = pd.DataFrame(forestscores, columns = columns)

df_forestscores_pivot = df_forestscores.pivot(index='min_samples_leaf', columns='max_depth', values=['auc'])
print(df_forestscores_pivot.round(3))
sns.heatmap(df_forestscores_pivot, annot=True, fmt=".3f")


# In[110]:


forestscores = []
counter = 1
for estimators_ in [10]:
    for depth_ in [2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        model =  RandomForestClassifier(n_estimators = estimators_, max_depth = depth_, min_samples_leaf = 250 , random_state = 42)
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_valid)[:,1]
        auc = roc_auc_score(y_valid, preds)
        forestscores.append((estimators_, depth_, auc))
        print(f'*************** {counter} forests completed ***************')
        counter = counter +1
        
columns = ['estimators', 'max_depth', 'auc']
df_forestscores = pd.DataFrame(forestscores, columns = columns)

df_forestscores_pivot = df_forestscores.pivot(index='estimators', columns='max_depth', values=['auc'])
print(df_forestscores_pivot.round(3))
sns.heatmap(df_forestscores_pivot, annot=True, fmt=".3f")            


# In[ ]:


forestscores = []
counter = 1
for estimators_ in [10,50,100,250]:
    for depth_ in [9]:
        model =  RandomForestClassifier(n_estimators = estimators_, max_depth = depth_, min_samples_leaf = 250 , random_state = 42)
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_valid)[:,1]
        auc = roc_auc_score(y_valid, preds)
        forestscores.append((estimators_, depth_, auc))
        print(f'*************** {counter} forests completed ***************')
        counter = counter +1
        
columns = ['estimators', 'max_depth', 'auc']
df_forestscores = pd.DataFrame(forestscores, columns = columns)

df_forestscores_pivot = df_forestscores.pivot(index='estimators', columns='max_depth', values=['auc'])
print(df_forestscores_pivot.round(3))
sns.heatmap(df_forestscores_pivot, annot=True, fmt=".3f")   


# In[8]:


finalRandomForest = RandomForestClassifier(n_estimators = 50,max_depth = 9, min_samples_leaf = 100, random_state = 42)
finalRandomForest.fit(x_full_train, y_full_train)    


# ## XGBoost
# 
# Parameters to tune: 
#     - eta
#     - max depth
#     - min child weight

# In[18]:


features = x_train.columns
dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(x_valid, label=y_valid, feature_names=features)
dfullTrainval = xgb.DMatrix(x_full_train, label=y_full_train, feature_names=features)
dTest = xgb.DMatrix(x_test, label = y_test, feature_names = features)


# In[11]:


get_ipython().run_cell_magic('capture', 'output', "scores = {}\n\nxgb_params = {\n    'eta': 0.01, \n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nwatchlist = [(dtrain, 'train'), (dval, 'val')]\n\ndef parse_xgb_output(output):\n    results = []\n\n    for line in output.stdout.strip().split('\\n'):\n        it_line, train_line, val_line = line.split('\\t')\n\n        it = int(it_line.strip('[]'))\n        train = float(train_line.split(':')[1])\n        val = float(val_line.split(':')[1])\n\n        results.append((it, train, val))\n    \n    columns = ['num_iter', 'train_auc', 'val_auc']\n    df_results = pd.DataFrame(results, columns=columns)\n    return df_results\n")


# ###  eta

# In[ ]:


scores = {}


# In[ ]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.01, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'eta=%s' % (xgb_params['eta'])\nscores[key] = parse_xgb_output(output)")


# In[161]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.05, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'eta=%s' % (xgb_params['eta'])\nscores[key] = parse_xgb_output(output)")


# In[162]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.1, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'eta=%s' % (xgb_params['eta'])\nscores[key] = parse_xgb_output(output)")


# In[163]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'eta=%s' % (xgb_params['eta'])\nscores[key] = parse_xgb_output(output)")


# In[ ]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 1.0, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'eta=%s' % (xgb_params['eta'])\nscores[key] = parse_xgb_output(output)")


# In[176]:


for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)

#plt.ylim(0.84, 0.88)
plt.legend()


# ###  max_depth

# In[178]:


scoresDepth = {}


# In[179]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 2,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'max_depth=%s' % (xgb_params['max_depth'])\nscoresDepth[key] = parse_xgb_output(output)")


# In[180]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 4,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'max_depth=%s' % (xgb_params['max_depth'])\nscoresDepth[key] = parse_xgb_output(output)")


# In[181]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'max_depth=%s' % (xgb_params['max_depth'])\nscoresDepth[key] = parse_xgb_output(output)")


# In[182]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 8,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'max_depth=%s' % (xgb_params['max_depth'])\nscoresDepth[key] = parse_xgb_output(output)")


# In[183]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 10,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'max_depth=%s' % (xgb_params['max_depth'])\nscoresDepth[key] = parse_xgb_output(output)")


# In[184]:


for max_depth, df_score in scoresDepth.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)

#plt.ylim(0.84, 0.88)
plt.legend()


# ## min child weight

# In[185]:


scoresChild = {}


# In[186]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'min_child_weight=%s' % (xgb_params['min_child_weight'])\nscoresChild[key] = parse_xgb_output(output)")


# In[187]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 10,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'min_child_weight=%s' % (xgb_params['min_child_weight'])\nscoresChild[key] = parse_xgb_output(output)")


# In[ ]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 30,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'min_child_weight=%s' % (xgb_params['min_child_weight'])\nscoresChild[key] = parse_xgb_output(output)")


# In[190]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1\n    'max_depth': 6,\n    'min_child_weight': 50,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\nkey = 'min_child_weight=%s' % (xgb_params['min_child_weight'])\nscoresChild[key] = parse_xgb_output(output)")


# In[192]:


for max_child, df_score in scoresChild.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_child)

#plt.ylim(0.84, 0.88)
plt.legend()


# ### final XGBOOST

# In[19]:


xgb_params = {
    'eta': 0.3, #try values: 0.01, 0.05, 0.1, 0.3, 1
    'max_depth': 6,
    'min_child_weight': 30,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
finalXGBoost = xgb.train(xgb_params, dfullTrainval, num_boost_round=25,
                  verbose_eval=5,
                  evals=watchlist)


# # final evaluation of models on test set
# 
# ## test scoring

# In[23]:


logisticTest = finalLogisticReg.predict_proba(x_test[['revolveUtilUnsecured', 'times30to59dayslate', 'times60to89dayslate', 'times90DaysLate']])[:,1]
decisionTreeTest = finalDeicisionTree.predict_proba(x_test)[:,1]
randomForestTest = finalRandomForest.predict_proba(x_test)[:,1]
xgbTest = finalXGBoost.predict(dTest)


# ## AUC calculation

# In[28]:


logisticAuc = roc_auc_score(y_test, logisticTest)
decistionTreeAuc = roc_auc_score(y_test, decisionTreeTest)
randomforestAuc = roc_auc_score(y_test, randomForestTest)
xgbAuc = roc_auc_score(y_test, xgbTest)

print(f'Model: logistic regression, AUC: {logisticAuc}')
print(f'Model: decision tree, AUC: {decistionTreeAuc}')
print(f'Model: random forest, AUC: {randomforestAuc}')
print(f'Model: xgBoost, AUC: {xgbAuc}')
      


# ### random forest with no missing impute

# In[32]:


data.revolveUtilUnsecured = np.log1p(data.revolveUtilUnsecured)
data.age[data.age < 18] = 18
data.times30to59dayslate = np.log1p(data.times30to59dayslate)

#data.loc[data.monthlyIncome.isnull(), 'incomeNullInd'] = 1
#data.incomeNullInd = data.incomeNullInd.fillna(0).astype(int)

incomeAvg = data.monthlyIncome.mean()

data.totalCreditLines = np.log1p(data.totalCreditLines)
data.times90DaysLate = np.log1p(data.times90DaysLate)
data.realEstateLines = np.log1p(data.realEstateLines)
data.times60to89dayslate = np.log1p(data.times60to89dayslate)


df_full_train, df_test = train_test_split(data, test_size=0.20, random_state=42)
df_train, df_valid = train_test_split(df_full_train, test_size = 0.25, random_state = 42)

x_train = df_train
x_valid = df_valid
x_test = df_test
x_full_train = df_full_train

y_train = x_train.delinquent.values
y_valid = x_valid.delinquent.values
y_test = x_test.delinquent.values
y_full_train = x_full_train.delinquent.values

del x_train['delinquent']
del x_valid['delinquent']
del x_test['delinquent']
del x_full_train['delinquent']


# In[33]:


finalRandomForest = RandomForestClassifier(n_estimators = 50,max_depth = 9, min_samples_leaf = 100, random_state = 42)
finalRandomForest.fit(x_full_train, y_full_train)    


# In[34]:


randomForestTest = finalRandomForest.predict_proba(x_test)[:,1]
randomforestAuc = roc_auc_score(y_test, randomForestTest)
print(f'Model: random forest, no missing imputation, AUC: {randomforestAuc}')

