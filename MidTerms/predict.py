  
import pickle

from flask import Flask
from flask import request
from flask import jsonify

import numpy as np
import pandas as pd


model_file = 'model_rf.bin'

with open(model_file, 'rb') as f_in:
    incomeAvg, model = pickle.load(f_in)
    
app = Flask('midTerm')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    customer = pd.DataFrame(customer, columns = ["revolveUtilUnsecured","age","times30to59dayslate","debtRatio","monthlyIncome","totalCreditLines","times90DaysLate","realEstateLines",    "times60to89dayslate",
    "dependents"], index = [0])
    customer.revolveUtilUnsecured = np.log1p(customer.revolveUtilUnsecured)
    customer.age[customer.age < 18] = 18
    customer.times30to59dayslate = np.log1p(customer.times30to59dayslate)

    customer.loc[customer.monthlyIncome.isnull(), 'incomeNullInd'] = 1
    customer.incomeNullInd = customer.incomeNullInd.fillna(0).astype(int)
    
    customer.monthlyIncome.fillna(incomeAvg, inplace = True)

    customer.totalCreditLines = np.log1p(customer.totalCreditLines)
    customer.times90DaysLate = np.log1p(customer.times90DaysLate)
    customer.realEstateLines = np.log1p(customer.realEstateLines)
    customer.times60to89dayslate = np.log1p(customer.times60to89dayslate)
    customer.dependents.fillna(0,inplace = True)
    
    
    
    y_pred = model.predict_proba(customer.values)[0, 1]
 
    result = {
        'delinquency_probability': float(y_pred),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

