from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define the Flask API route
@app.route('/analyze_profile', methods=['POST'])
def analyze_profile():
    # Receive input data from the client
    data = request.json

    # Extract input values
    username = data.get('username')
    full_name = data.get('fullName')
    description = data.get('description')
    profile_pic = data.get('profilePic')
    post_count = data.get('postCount')
    followers_count = data.get('followersCount')
    following_count = data.get('followingCount')
    private_account = data.get('privateAccount')

    # Create a dictionary containing the generated outputs
    outputs = {
        'profilePic': profile_pic,
        'usernameLength': len(username),
        'fullNameWords': len(full_name.split()),
        'nameEqualsUsername': username == full_name,
        'descriptionLength': len(description),
        'externalUrlPresent': 'https://' in description or 'http://' in description,
        'postCount': post_count,
        'followersCount': followers_count,
        'followingCount': following_count,
        'privateAccount': private_account,
    }

    # Call the model prediction function with the generated outputs
    model_predictions = predict(np.array([list(outputs.values())]))

    # Return the model predictions as a JSON response
    return jsonify(model_predictions)