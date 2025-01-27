import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['strength'] = df['strength'].replace({0: 'Weak', 1: 'Medium', 2: 'Strong'})
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
def preprocess_data(df):
    passwords = df['password'].values
    labels = df['strength'].values
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(passwords)
    sequences = tokenizer.texts_to_sequences(passwords)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    return padded_sequences, labels_categorical, tokenizer, max_len, label_encoder
def build_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
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
def train_and_evaluate(X_train, y_train, X_test, y_test, vocab_size, max_len):
    model = build_model(vocab_size, max_len)
    print("Training the model...")
    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32)
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return model
def predict_password_strength(model, tokenizer, label_encoder, max_len, password):
    sequence = tokenizer.texts_to_sequences([password])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded_sequence)
    strength = label_encoder.inverse_transform([np.argmax(prediction)])
    score, recommendations = analyze_password(password)
    return strength[0], score, recommendations
if __name__ == "__main__":
    dataset_path = r""  # Replace with your dataset path
    print("Loading dataset...")
    df = load_data(dataset_path)
    if df is None:
        exit()
    print("Preprocessing data...")
    X, y, tokenizer, max_len, label_encoder = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vocab_size = len(tokenizer.word_index) + 1
    model = train_and_evaluate(X_train, y_train, X_test, y_test, vocab_size, max_len)
    model.save("password_strength_model.h5")
    with open("tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())
    print("Model and tokenizer saved.")
    print("Enter a password to test its strength:")
    sample_password = input("> ")
    strength, score, recommendations = predict_password_strength(
        model, tokenizer, label_encoder, max_len, sample_password
    )
    print(f"The password strength is: {strength}")
    print(f"Password Score: {score}/100")
    if recommendations:
        print("Recommendations to improve your password:")
        for rec in recommendations:
            print(f"- {rec}")

