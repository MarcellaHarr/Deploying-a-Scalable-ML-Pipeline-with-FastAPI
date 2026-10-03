# == Import libraries ==
import json

import requests

# == send GET request for the root endpoint ==
r = requests.get(
    "http://127.0.0.1:8000"
)

# == print status code ==
print(
    f"GET Status Code: {r.status_code}"
)
# == print welcome message ==
print(
    f"Message: {r.json()['message']}"
)


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# == send POST request for the data endpoint ==
r = requests.post(
    "http://127.0.0.1:8000/data/",
    data=json.dumps(data)
)

# == print status code ==
print(
    f"POST Status Code: {r.status_code}"
)
# == print prediction result ==
print(
    f"Predictive Result: {r.json()['result']}"
)
