import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Rosa_rubiginosa_1.jpg/220px-Rosa_rubiginosa_1.jpg'}

result = requests.post(url, json=data).json()
print(result)
