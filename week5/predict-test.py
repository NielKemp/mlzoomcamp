#import packages
import pickle


model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as model_in:
    model = pickle.load(model_in)
    
with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)
    
    
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}
X = dv.transform([customer])
pred = model.predict_proba(X)
print(f'probability of customer churning: {pred}')
