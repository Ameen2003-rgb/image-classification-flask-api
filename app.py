from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from flask_httpauth import HTTPBasicAuth
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session handling
auth = HTTPBasicAuth()

# Define username and password
USERNAME = "admin"
PASSWORD = "password123"

@auth.verify_password
def verify_password(username, password):
    return username == USERNAME and password == PASSWORD

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Define class labels
class_labels = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

@app.route('/')
def index():
    if 'logged_in' in session:
        return render_template('upload.html')  # Show upload form after login
    return render_template('index.html')  # Show login page first

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if verify_password(username, password):
        session['logged_in'] = True
        return jsonify({"success": True})
    
    return jsonify({"success": False}), 401

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
@auth.login_required  # 🔒 Require authentication
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Process the image
    image = Image.open(file).resize((32, 32))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]  # Get class name

    return jsonify({"predicted_class": predicted_label})  # Return JSON response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

