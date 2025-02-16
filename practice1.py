import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras import layers
nltk.download('punkt')
nltk.download('wordnet')
data_file=open('intent.json').read()
intents=json.loads(data_file)
lemmatizer = WordNetLemmatizer()
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
all_words = sorted(set(all_words))
classes = sorted(classes)

X_train = []
y_train = []

for sentence in training_sentences:
    bag = [1 if word in sentence else 0 for word in all_words]
    X_train.append(bag)

for label in training_labels:
    y_train.append(classes.index(label))

X_train = np.array(X_train)
y_train = np.array(y_train)


model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(len(X_train[0]),)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1)
def preprocess_input(user_input):
    """Converts user input into a bag-of-words vector."""
    user_words = nltk.word_tokenize(user_input)
    user_words = [lemmatizer.lemmatize(word.lower()) for word in user_words]
    bag = [1 if word in user_words else 0 for word in all_words]
    return np.array(bag)

def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Bot: Goodbye! Take care and remember, I'm always here for you!")
            break
        
        
        user_bag = preprocess_input(user_input)
        
        prediction = model.predict(np.array([user_bag]), verbose=0)
        response_index = np.argmax(prediction)
        confidence = prediction[0][response_index]
        
        if confidence > 0.7:  
            response_tag = classes[response_index]

            for intent in intents['intents']:
                if intent['tag'] == response_tag:
                    response = random.choice(intent['responses'])
                    print(f"Bot: {response}")
                    
                    
                    if response_tag in ["sad", "depressed", "anxious", "stressed", "worthless"]:
                        user_choice = input(f"Bot: Would you like some tips to reduce {response_tag.replace('_', ' ')}? (yes/no) ")
                        if user_choice.lower() == "yes":
                            if response_tag == "sad":
                                print("Bot: Tips for sadness:\n1. Talk to someone you trust.\n2. Engage in activities you enjoy.\n3. Practice gratitude journaling.")
                            elif response_tag == "depressed":
                                print("Bot: Tips for depression:\n1. Reach out to a therapist.\n2. Take small steps toward self-care.\n3. Practice mindfulness and deep breathing.")
                            elif response_tag == "anxious":
                                print("Bot: Tips for anxiety:\n1. Try deep breathing exercises.\n2. Write down your worries.\n3. Exercise regularly.")
                            elif response_tag == "stressed":
                                print("Bot: Tips for stress relief:\n1. Take short breaks.\n2. Practice mindfulness or yoga.\n3. Set small, achievable goals.")
                            elif response_tag == "worthless":
                                print("Bot: Tips to feel more worthy:\n1. Write down your achievements, big or small.\n2. Focus on self-compassion.\n3. Talk to a supportive friend or therapist.")
                        else:
                            print("Bot: That's okay. I'm here if you need anything.")
        else:
            print("Bot: I'm sorry, I didn't understand that. Could you rephrase?")

if __name__ == "__main__":
    chat()
