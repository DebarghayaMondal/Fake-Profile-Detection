import requests

# Define the input data as a dictionary
input_data = {
    'username': '...',
    'fullName': '...',
    'description': '...',
    'profilePic': '...',
    'postCount': ...,
    'followersCount': ...,
    'followingCount': ...,
    'privateAccount': ...
}

# Send a POST request to the Flask API
response = requests.post('http://localhost:5000/analyze_profile', json=input_data)

# Parse and print the model predictions
model_predictions = response.json()
result=predict(model_predictions)
print(result)        