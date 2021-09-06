#import packages
import numpy as np
import pandas as pd

#question 1 
npV = np.__version__
print(f"Version of numpy is: {npV}")

#question 2
pdV = pd.__version__
print(f"Version of pandas is: {pdV}")

#question 3 
data = pd.read_csv("../input/data.csv")

print("Average price of BMW's: " + np.mean(data[data['Make'] =='BMW']['MSRP']).astype(str))

#question 4
missings = data[data['Year'] >= 2015]['Engine HP'].isnull().sum()
print(f"Number of missing Engine HP entries for models > 2015: {missings}")

#question 5
avgEngineHP = data['Engine HP'].mean()
print(f"Average Engine HP: {avgEngineHP}")

data.loc[:, 'Engine HP'] = data['Engine HP'].fillna(avgEngineHP)
avgEngineHP2 =  data['Engine HP'].mean()

print(f"Average Engine HP after impute: {avgEngineHP2}")

#question 6
subset = data[data['Make'] == 'Rolls-Royce'][['Engine HP', 'Engine Cylinders', 'highway MPG']].drop_duplicates()
X = subset.values
XTX = np.matmul(X.T, X)
invXTX = np.linalg.inv(XTX)

print("Total of matrix elements of inverse of XTX: " + (np.sum(np.linalg.inv(np.matmul(X.T,X)))).astype(str))

#question 7
y = [1000, 1100, 900, 1200, 1000, 850, 1300]
w = np.matmul(np.matmul(invXTX, X.T), y)

print(f"Resultant weights (w): {w}")

