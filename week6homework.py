#apply starter code
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import math

import xgboost as xgb
columns = [
    'neighbourhood_group', 'room_type', 'latitude', 'longitude',
    'minimum_nights', 'number_of_reviews','reviews_per_month',
    'calculated_host_listings_count', 'availability_365',
    'price'
]

df = pd.read_csv('AB_NYC_2019.csv', usecols=columns)
#fill NAs
df.reviews_per_month = df.reviews_per_month.fillna(0)

#Apply log transform to price
df.price = np.log1p(df.price)

#train validate test split

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_valid = train_test_split(df_full_train, test_size = 0.25, random_state = 1)

y_train = df_train.price.values
y_valid = df_valid.price.values
y_test = df_test.price.values

del df_train['price']
del df_valid['price']
del df_test['price']

#convert dataframe into matrices
train_dicts = df_train.to_dict(orient = 'records')

dv = DictVectorizer(sparse = False)
x_train = dv.fit_transform(train_dicts)

print("==================== QUESTION 1 ====================")
#Question 1 Model
dt = DecisionTreeRegressor(max_depth = 2)
dt.fit(x_train, y_train)
print(export_text(dt, feature_names=dv.get_feature_names()))

print("==================== QUESTION 2 ====================")
#Question 2 Model
valid_dicts = df_valid.to_dict(orient = 'records')
x_valid = dv.transform(valid_dicts)

dt2 = RandomForestRegressor(n_estimators = 10, random_state = 1, n_jobs = -1)
dt2.fit(x_train, y_train)

predVal = dt2.predict(x_valid)
rmse = math.sqrt(mean_squared_error(y_valid, predVal))
print(f"rmse on validation model is {rmse}")

print("==================== QUESTION 3 ====================")
#question 3
for n in np.linspace(10,200, 20).astype(int):
    dt3= RandomForestRegressor(n_estimators = n, random_state = 1, n_jobs = -1)
    dt3.fit(x_train, y_train)
    predVal = dt3.predict(x_valid)
    rmse = math.sqrt(mean_squared_error(y_valid, predVal))
    print(f"n_estimators = {n},    rmse = {rmse}")
    

print("==================== QUESTION 4 ====================")    
#question 4
scores = []
for depth in [10,15,20,25]:
    for n in np.linspace(10,200, 20).astype(int):
        dt3= RandomForestRegressor(n_estimators = n, max_depth = depth, random_state = 1, n_jobs = -1)
        dt3.fit(x_train, y_train)
        predVal = dt3.predict(x_valid)
        rmse = math.sqrt(mean_squared_error(y_valid, predVal))
        scores.append((depth, n, rmse))

q4Scores = pd.DataFrame(scores, columns = ['max_depth', 'n_estimators', 'rmse'])
q4ScoresPivot = q4Scores.pivot_table(index = ['n_estimators'], columns = ['max_depth'], values = ['rmse'])
print(q4ScoresPivot.round(3))


print("==================== QUESTION 5 ====================")   
#question 5
dt5 = RandomForestRegressor(n_estimators = 10, max_depth = 20, random_state = 1, n_jobs = -1)
dt5.fit(x_train, y_train)
imps = dt5.feature_importances_.round(8)

dfImps = pd.DataFrame(imps, index = dv.feature_names_)
print(dfImps)


print("==================== QUESTION 6 ====================")   
#question 6
features = dv.get_feature_names()
dtrain = xgb.DMatrix(x_train, label = y_train, feature_names = features)
dval = xgb.DMatrix(x_valid, label = y_valid, feature_names = features)
watchlist = [(dtrain, 'train'), (dval, 'val')]

xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
print("eta: 0.3")
model1 = xgb.train(xgb_params, dtrain, num_boost_round = 100, verbose_eval = 5, evals=watchlist)
y_pred1 = model1.predict(dval)
rmse1 = math.sqrt(mean_squared_error(y_valid, y_pred1))


xgb_params = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
print("eta: 0.1")
model2 = xgb.train(xgb_params, dtrain, num_boost_round = 100, verbose_eval = 5, evals=watchlist)
y_pred2 = model2.predict(dval)
rmse2 = math.sqrt(mean_squared_error(y_valid, y_pred2))


xgb_params = {
    'eta': 0.01, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
print("eta: 0.01")
model3 = xgb.train(xgb_params, dtrain, num_boost_round = 100, verbose_eval = 5, evals=watchlist)
y_pred3 = model3.predict(dval)
rmse3 = math.sqrt(mean_squared_error(y_valid, y_pred3))

print(f"final validation rmse for ema = 0.3, rmse = {rmse1}")
print(f"final validation rmse for ema = 0.1, rmse = {rmse2}")
print(f"final validation rmse for ema = 0.01, rmse = {rmse3}")

