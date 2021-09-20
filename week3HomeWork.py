 #import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

import math
import numpy as np

 
#import data 
df = pd.read_csv("../input/AB_NYC_2019.csv")
df = df.fillna(0)


features = ['neighbourhood_group', 'room_type', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']


###question 1
print('Mode is: ' + df.neighbourhood_group.mode().astype(str))


###question 2

##split data
df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
df_train, df_val = train_test_split(df_train, test_size = 0.25, random_state = 42)



x_train = df_train[features].reset_index(drop = True)
x_test  = df_test[features].reset_index(drop = True)
x_val   = df_val[features].reset_index(drop = True)

y_train = df_train.price.values
y_test = df_test.price.values
y_val = df_val.price.values

print(df.shape)
print(df_train.shape)
print(df_test.shape)
print(df_val.shape)
print()
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print()
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

print(x_train.dtypes)
print(x_train.corr())
print('number of reviews and reviews per month')

### question 3
print('mutual info with neighbourhood group: ')
print(round(mutual_info_score(y_train, x_train.neighbourhood_group),2))
print(' ')
print('mutual info with room type: ')
print(round(mutual_info_score(y_train, x_train.room_type),2))

### question 4
##binarize target
above_average_train = (y_train >= 152).astype(int)
above_average_val   = (y_val >= 152).astype(int)
above_average_test  = (y_test >= 152).astype(int)

##onehotencoding
#init ohe on training data
train_dict = x_train.to_dict(orient = 'records')
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
x_train_ohe = dv.fit_transform(train_dict)

#fit on validation data
val_dict = x_val.to_dict(orient='records')
x_val_ohe = dv.transform(val_dict)



model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(x_train_ohe, above_average_train)
val_preds = model.predict(x_val_ohe)
acc_ = accuracy_score(above_average_val,val_preds)
print(round(accuracy_score(above_average_val,val_preds),2))

accs = []
accs.append(['global acc', acc_])
### question 5
for col in features:
    temp = x_train.copy()
    temp_val = x_val.copy()
    
    del temp[col]
    del temp_val[col]
    
    #init ohe on training data
    temp_train_dict = temp.to_dict(orient = 'records')
    temp_dv = DictVectorizer(sparse=False)
    temp_dv.fit(temp_train_dict)
    train_temp_ohe = dv.fit_transform(temp_train_dict)

    #fit on validation data
    temp_val_dict = temp_val.to_dict(orient='records')
    val_temp_ohe = dv.transform(temp_val_dict)
    
    tempModel = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
    tempModel.fit(train_temp_ohe,above_average_train)
    
    tempVal = tempModel.predict(val_temp_ohe)
    tempAcc = accuracy_score(above_average_val,tempVal)
    
    accs.append([col, tempAcc])
    
print(accs)

### question 6
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

rmses = []

for alpha_ in [0,0.01,0.1,1,10]:
    tempModel = Ridge(alpha = alpha_)
    tempModel.fit(x_train_ohe, y_train_log)
    
    tempVal = tempModel.predict(x_val_ohe)
    rmse = math.sqrt(mean_squared_error(y_val_log, tempVal))
    
    rmses.append([alpha_, round(rmse,3)])
    
print(rmses) 	
