
import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'
customer = {
    "revolveUtilUnsecured": 0.877151,
    "age": 29,
    "times30to59dayslate": 0,
    "debtRatio": 0.561879,
    "monthlyIncome": 8000,
    "totalCreditLines": 4,
    "times90DaysLate": 0,
    "realEstateLines": 4,
    "times60to89dayslate": 0,
    "dependents": 2.0
}


response = requests.post(url, json=customer).json()
print(response)

 #delinquent  revolveUtilUnsecured  age  times30to59dayslate  debtRatio  monthlyIncome  totalCreditLines  times90DaysLate  realEstateLines  times60to89dayslate  dependents
 #       1              0.766127   45                    2   0.802982         9120.0                13                0                6                    0         2.0
 #       0              0.957151   40                    0   0.121876         2600.0                 4                0                0                    0         1.0
 #       0              0.658180   38                    1   0.085113         3042.0                 2                1                0                    0         0.0
 #       0              0.233810   30                    0   0.036050         3300.0                 5                0                0                    0         0.0
  #      0              0.907239   49                    1   0.024926        63588.0                 7                0                1                    0         0.0

