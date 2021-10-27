#import packages
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


#read in data
data = pd.read_csv('data.csv')
del data['Unnamed: 0']



data.rename(columns={'SeriousDlqin2yrs':'delinquent','RevolvingUtilizationOfUnsecuredLines':'revolveUtilUnsecured','NumberOfTime30-59DaysPastDueNotWorse':'times30to59dayslate','DebtRatio':'debtRatio', 'MonthlyIncome':'monthlyIncome', 'NumberOfOpenCreditLinesAndLoans':'totalCreditLines', 'NumberOfTimes90DaysLate':'times90DaysLate', 'NumberRealEstateLoansOrLines':'realEstateLines', 'NumberOfTime60-89DaysPastDueNotWorse':'times60to89dayslate', 'NumberOfDependents':'dependents'   }, inplace = True)

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




# final model
finalRandomForest = RandomForestClassifier(n_estimators = 50,max_depth = 9, min_samples_leaf = 100, random_state = 42)
finalRandomForest.fit(x_full_train, y_full_train)    


# check final model performance on test set
randomForestTest = finalRandomForest.predict_proba(x_test)[:,1]
randomforestAuc = roc_auc_score(y_test, randomForestTest)
print(f'Model: random forest, AUC: {randomforestAuc}')


#write model to model.bin file
output_file = 'model_rf.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((incomeAvg, finalRandomForest), f_out)

print(f'the model is saved to {output_file}')
