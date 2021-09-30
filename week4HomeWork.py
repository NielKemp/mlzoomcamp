#import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from tqdm.auto import tqdm


###################################################################################################
# predefined preperation
###################################################################################################
df = pd.read_csv('../input/CreditScoring.csv')
df.columns = df.columns.str.lower()


#define mappings
status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

#apply mappings to decode numerics
df.status = df.status.map(status_values)
df.home = df.home.map(home_values)
df.marital = df.marital.map(marital_values)
df.records = df.records.map(records_values)
df.job = df.job.map(job_values)


#prepare numerical variables
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=0)
    
#remove clients with unknown default status
df = df[df.status != 'unk'].reset_index(drop=True)

#create target variable
df['default'] = (df.status == 'default').astype(int)
del df['status']

###################################################################################################
###################################################################################################
###################################################################################################

print(df.dtypes)

###################################################################################################

#split into train/test/split
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=1)


###################################################################################################
##Question 1
###################################################################################################
numericalCols = [c for c in df_train.select_dtypes(include=np.number) if c not in ('default')]

for col in numericalCols:
    if metrics.roc_auc_score(df_train.default,df_train[col]) < 0.5:
        print("column: " + col + " \t\t  auc: " + metrics.roc_auc_score(df_train.default,-df_train[col]).astype(str))
    else:
        print("column: " + col + " \t\t  auc: " + metrics.roc_auc_score(df_train.default,df_train[col]).astype(str))
        
        
###################################################################################################
##Question 2
###################################################################################################
features = ['seniority', 'income', 'assets', 'records', 'job', 'home']

x_train_reduce = df_train[features]
y_train = df_train.default.values

x_val_reduce = df_val[features]
y_val = df_val.default.values

train_dict = x_train_reduce.to_dict(orient = 'records')
val_dict = x_val_reduce.to_dict(orient = 'records')


dv = DictVectorizer(sparse=False)
x_train_reduce_ohe = dv.fit_transform(train_dict)

x_val_reduce_ohe = dv.transform(val_dict)

model = LogisticRegression(solver = 'liblinear', C=1.0, max_iter=1000)
model.fit(x_train_reduce_ohe, y_train)

y_val_preds = model.predict_proba(x_val_reduce_ohe)[:,1]

print(metrics.roc_auc_score(y_val, y_val_preds))



###################################################################################################
##Question 3 & 4
###################################################################################################

thresholds = np.linspace(0,1,101)
chartdat = []
print(thresholds)

for t_ in thresholds:
    act_pos = (y_val == 1)
    act_neg = (y_val == 0)
    pred_pos = (y_val_preds >= t_)
    pred_neg = (y_val_preds < t_)
    
    tp = (pred_pos & act_pos).sum()
    tn = (pred_neg & act_neg).sum()
    
    fp = (pred_pos & act_neg).sum()
    fn = (pred_neg & act_pos).sum()
    
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r/(p+r)
    
    chartdat.append([t_,p,r,f1])
    
df_scores = pd.DataFrame(chartdat, columns = ['threshold', 'p', 'r','f'])
print(df_scores)
plt.plot(df_scores.threshold, df_scores.p, label = 'precision')
plt.plot(df_scores.threshold, df_scores.r, label = 'recall')
plt.plot(df_scores.threshold, df_scores.f, label = 'f1-score')
#plt.show()

###################################################################################################
##Question 5
###################################################################################################

def modelTrain(train, target, C = 1.0):

    train_dict = train.to_dict(orient = 'records')
    dv = DictVectorizer(sparse=False)
    train_ohe = dv.fit_transform(train_dict)
    
    model = LogisticRegression(solver = 'liblinear', C=C, max_iter=1000)
    model.fit(train_ohe, target)
    
    return dv, model
    
def predict(val, dv, model):
    val_dict = val.to_dict(orient = 'records')
    val_ohe = dv.transform(val_dict)
    
    preds = model.predict_proba(val_ohe)[:,1]
    return preds

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
aucs = []
counter = 1

for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
    print(f'Iter :  {counter}  start')
    fold_train = df_full_train.iloc[train_idx]
    fold_val = df_full_train.iloc[val_idx]
    print('fold of train/test built')
    
    dv, model = modelTrain(fold_train[features], fold_train.default, 1.0)
    print('model trained')
    scores = predict(fold_val[features], dv, model)
    print('model scored')
    
    auc = metrics.roc_auc_score(fold_val.default, scores)
    
    print(auc)
    print('auc calculated')
    aucs.append([auc])
    print('auc appended')
    print(f'Iter :  {counter}  end')
    counter = counter+1
        

print(np.mean(aucs))
print(np.std(aucs))

###################################################################################################
##Question 6
###################################################################################################

for c_ in [0.01, 0.1, 1, 10]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    Cscores = []
    for train_idx, val_idx in tqdm(kfold.split(df_full_train)):

        fold_train = df_full_train.iloc[train_idx]
        fold_val = df_full_train.iloc[val_idx]

    
        dv, model = modelTrain(fold_train[features], fold_train.default, c_)

        scores = predict(fold_val[features], dv, model)

    
        auc = metrics.roc_auc_score(fold_val.default, scores)
    
        Cscores.append([auc])
    m_ = np.mean(Cscores)
    s_ = np.std(Cscores)

    print('C=%s %.3f +- %.3f' % (c_, np.mean(Cscores), np.std(Cscores)))

