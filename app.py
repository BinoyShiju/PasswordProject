from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder
import hashlib
import requests
app = Flask(__name__)
model = load_model('password_strength_model.h5')
with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())
label_encoder = LabelEncoder()
label_encoder.fit(['Weak', 'Medium', 'Strong'])  
def analyze_password(password):
    length = len(password)
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()-_=+[]{};:'\",.<>?/`~" for c in password)
    score = length * 5
    if has_upper: score += 10
    if has_lower: score += 10
    if has_digit: score += 10
    if has_special: score += 15
    if length <= 5 and not (has_upper or has_digit or has_special):
        return score, ["This password is too simple and considered Weak."]
    recommendations = []
    if length < 8:
        recommendations.append("Increase the password length to at least 8 characters.")
    if not has_upper:
        recommendations.append("Include at least one uppercase letter.")
    if not has_lower:
        recommendations.append("Include at least one lowercase letter.")
    if not has_digit:
        recommendations.append("Include at least one numeric digit.")
    if not has_special:
        recommendations.append("Include at least one special character (e.g., !, @, #, $).")
    return score, recommendations
def check_password_leaked(password):
    sha1_hash = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
    first5 = sha1_hash[:5]
    rest_of_hash = sha1_hash[5:]
    url = f"https://api.pwnedpasswords.com/range/{first5}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        if rest_of_hash in response.text:
            return True  # Leaked
        return False  # Not leaked
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False
def predict_password_strength(password):
    sequence = tokenizer.texts_to_sequences([password])
    max_len = 16
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded_sequence)
    strength = label_encoder.inverse_transform([np.argmax(prediction)])
    score, recommendations = analyze_password(password)
    is_leaked = check_password_leaked(password)
    return strength[0], score, recommendations, is_leaked
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/analyze', methods=['POST'])
def analyze():
    password = request.form['password']
    strength, score, recommendations, is_leaked = predict_password_strength(password)
    return jsonify({
        'strength': strength,
        'score': score,
        'recommendations': recommendations,
        'is_leaked': is_leaked
    })
if __name__ == '__main__':
    app.run(debug=True)
