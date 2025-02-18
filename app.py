import json
import random
import nltk
import numpy as np
from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
port = int(os.getenv("PORT", 10000))

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
lemmatizer = WordNetLemmatizer()
with open('intent.json', 'r') as file:
    intents = json.load(file)

# Preprocess data
training_sentences = []
training_labels = []
classes = []
all_words = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words = [lemmatizer.lemmatize(word.lower()) for word in word_list]
        training_sentences.append(words)
        training_labels.append(intent['tag'])
        all_words.extend(words)

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Remove duplicates and sort
all_words = sorted(set(all_words))
classes = sorted(classes)

# Prepare training data
X_train = []
y_train = []

for sentence in training_sentences:
    bag = [1 if word in sentence else 0 for word in all_words]
    X_train.append(bag)

for label in training_labels:
    y_train.append(classes.index(label))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Build model
model = keras.Sequential([
    keras.Input(shape=(len(X_train[0]),)),  # Define input shape
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1)  # Train the model

# Function to preprocess user input
def preprocess_input(user_input):
    """Converts user input into a bag-of-words vector."""
    user_words = nltk.word_tokenize(user_input)
    user_words = [lemmatizer.lemmatize(word.lower()) for word in user_words]
    bag = [1 if word in user_words else 0 for word in all_words]
    return np.array([bag])

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return jsonify({"message": "MoodMate Chatbot is running!"})

    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "Please provide a message"}), 400

    user_bag = preprocess_input(user_input)
    prediction = model.predict(user_bag, verbose=0)
    response_index = np.argmax(prediction)
    confidence = prediction[0][response_index]

    if confidence > 0.7:
        response_tag = classes[response_index]

        for intent in intents['intents']:
            if intent['tag'] == response_tag:
                response = random.choice(intent['responses'])
                return jsonify({"response": response})

    return jsonify({"response": "I'm sorry, I didn't understand that. Can you rephrase?"})

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
