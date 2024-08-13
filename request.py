import requests

# URL where your Flask app is running
url = 'http://localhost:5000/predict_api'

# Data to send in the POST request
data = {'experience': 'I have 2 years of experience in this field'}  # Modify this accordingly

# Send the POST request
r = requests.post(url, json=data)

# Print the response from the server
print(r.json())
