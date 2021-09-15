#import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


#import data
data = pd.read_csv('../input/AB_NYC_2019.csv')


#histogram with 50 bins
#sns.histplot(data.price, bins = 50)
#plt.show()
#histogram with 25 bins
#sns.histplot(data.price, bins = 25)
#plt.show()

features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

#Question 1
print(data[features].isnull().sum())
#reviews per month has missings: 10052

#Question 2
print(data.minimum_nights.median())
#median for minimum nights: 3

#========================================================================
#=================== setting up validation framework ====================
#========================================================================

#create counts of train/test/val splits
n = len(data)
n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test

#shuffle idx
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

#create test/train/val splist using randomized index
train = data.iloc[idx[:n_train]]
val   = data.iloc[idx[n_train:n_train+n_val]]
test  = data.iloc[idx[n_train+n_val:]]

#keep only features
x_train = train.loc[:,features]
x_val   = val.loc[:,features]
x_test  = test.loc[:,features]

#split out targets (price field)
y_train = np.log1p(train.price.values)
y_val = np.log1p(val.price.values)
y_test = np.log1p(test.price.values)


print("test, train val data created succesfully")
print(x_val.isnull().sum())


#define functions

#training function
def regutrainLin(data, y, r):
    #add 1's for W0
    ones = np.ones(data.shape[0])
    X = np.column_stack([ones, data])
    
  
    #compute normal equation: inv(XTX)XTy
    XTX = X.T.dot(X) + r * np.eye(X.shape[1])
    
    
    XTXInv = np.linalg.inv(XTX)
    
    w_0 = XTXInv.dot(X.T).dot(y)
    
    return w_0
   
def trainLin(X, y):
    #add 1's for W0
    ones = np.ones(X.shape[0])

    X = np.column_stack([ones, X])
    
  
    #compute normal equation: inv(XTX)XTy
    XTX = X.T.dot(X)
    
    XTXInv = np.linalg.inv(XTX)
    
    w_0 = XTXInv.dot(X.T).dot(y)
   
    return w_0
   
   
#scoring function    
def scoreLin(w, toScore):
   score = w[0] + toScore.dot(w[1:])
   return score 
    
    
#data prep function    
def prepX(X):
    temp = X.copy()
    temp.loc[:,'reviews_per_month']  = temp['reviews_per_month'].fillna(0)
    return temp

#data prep function    
def imputeMean(X):
    temp = X.copy()
    mean_ = temp.reviews_per_month.mean()
        
    temp.loc[:,'reviews_per_month']  = temp['reviews_per_month'].fillna(mean_)
    
    return mean_, temp

#rmse function
def rmse_(yhat, yact):
    se = (yact-yhat)**2
    return np.sqrt(se.mean())    
    
#=========================================================================================================
#========================================= Q3: different imputes =========================================
#=========================================================================================================
print("====================================== impute 0 ======================================")  
#impute missings    


x_train_imp_0 = prepX(x_train) 

q3_val_imp0 = prepX(x_val)  

weights = trainLin(x_train_imp_0, y_train)
score_val_imp0 = scoreLin(weights, q3_val_imp0)

print("validation rmse when impute with 0: " + rmse_(score_val_imp0, y_val).round(2).astype(str))   


print("====================================== impute mean ======================================")
impMean, x_train_impm = imputeMean(x_train)  
weights_imp = trainLin(x_train_impm, y_train)


x_val_imp = x_val.copy()
x_val_imp.loc[:,'reviews_per_month']  = x_val_imp['reviews_per_month'].fillna(impMean)
s_val_imp = scoreLin(weights_imp, x_val_imp)

print("valid rmse when impute with mean: "  + rmse_(s_val_imp, y_val).round(2).astype(str))   



#=======================================================================================================
#======================================= Q4: add reguralization ========================================
#=======================================================================================================
print("====================================== impute 0 and regu ======================================")
regOptions = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

for r in regOptions:
    x_train_imp0 = prepX(x_train)  
    weights = regutrainLin(x_train_imp0, y_train,r)
    
    x_val_imp0 = prepX(x_val)
    
    s_val_imp0 = scoreLin(weights, x_val_imp0)
    error = rmse_(s_val_imp0, y_val).round(2)  
    print(f"reg term = {r}, validation rmse = {error}")
    
    
#=======================================================================================================
#========================================= Q5: different seeds =========================================
#=======================================================================================================

seeds = [0,1,2,3,4,5,6,7,8,9]
rmseList = []

#create counts of train/test/val splits
n = len(data)
n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test

for s in seeds:
#shuffle idx
    idx = np.arange(n)
    np.random.seed(s)
    np.random.shuffle(idx)

#create test/train/val splist using randomized index
    train = data.iloc[idx[:n_train]]
    val   = data.iloc[idx[n_train:n_train+n_val]]
    test  = data.iloc[idx[n_train+n_val:]]

#keep only features
    x_train = train.loc[:,features]
    x_val   = val.loc[:,features]
    x_test  = test.loc[:,features]

#split out targets (price field)
    y_train = np.log1p(train.price.values)
    y_val = np.log1p(val.price.values)
    y_test = np.log1p(test.price.values)


    x_train_imp0 = prepX(x_train) 
    x_val_imp0 = prepX(x_val)
    
    weights = trainLin(x_train_imp0, y_train)
    
    
    sc = scoreLin(weights, x_val_imp0)
    error = rmse_(sc, y_val).round(2)  
    rmseList.append(error)
    
print(rmseList)
print('Standard dev of validation rmse scores: ' + np.std(rmseList).round(3).astype(str))

#=====================================================================================================
#========================================= Q6: train + valid =========================================
#=====================================================================================================

#create counts of train/test/val splits
n = len(data)
n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)

train_val = data.iloc[idx[:n_train+n_val]]
test  = data.iloc[idx[n_train+n_val:]]

#keep only features
x_train_val = train_val.loc[:,features]
x_test  = test.loc[:,features]

#split out targets (price field)
y_train_val = np.log1p(train_val.price.values)

y_test = np.log1p(test.price.values)


x_tv_0 = prepX(x_train_val) 
weights_tv = regutrainLin(x_tv_0, y_train_val,0.001)

x_test_0 = prepX(x_test)

sTest = scoreLin(weights_tv, x_test_0)

errorTest = rmse_(sTest, y_test).round(2)  
print(errorTest)  




