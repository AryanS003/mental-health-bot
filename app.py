import nltk
import sqlite3
import json
import random
import numpy as np
from flask import Flask, render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# SQLite database setup
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  session_id TEXT, 
                  user_message TEXT, 
                  bot_response TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Sample intents dataset
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"],
            "responses": [
                "Hello! I'm here to support you with your mental health. How can I help today?",
                "Hi there! It's great to hear from you. What's on your mind?",
                "Hey! I'm ready to listen. How are you feeling?"
            ]
        },
        {
            "tag": "feeling_sad",
            "patterns": ["i feel sad", "i'm depressed", "feeling down", "i'm so upset", "life feels heavy", "i can't stop crying"],
            "responses": [
                "I'm really sorry you're feeling this way. Would you like to share what's been going on?",
                "It's okay to feel sad sometimes. I'm here for you. Want to talk about what's making you feel down?",
                "I'm here to listen. Can you tell me more about how you're feeling?"
            ]
        },
        {
            "tag": "who_are_you",
            "patterns": ["who are you", "what are you", "who is this", "tell me about yourself", "what's your name", "who made you"],
            "responses": [
                "I'm a Mental Health Support Bot, created to provide empathetic support and resources for your mental well-being. I'm here to listen and help whenever you need me!",
                "Hi! I'm a friendly chatbot designed to assist with mental health concerns. My creators at xAI made me to offer support and guidance. What's on your mind?",
                "I'm your virtual companion for mental health support, built to listen and provide helpful responses. No name yet, but you can call me your friend! How can I assist you today?"
            ]
        },
        {
            "tag": "anxiety",
            "patterns": ["i feel anxious", "i'm nervous", "feeling stressed", "i'm overwhelmed", "i can't calm down", "panic attack"],
            "responses": [
                "Anxiety can be really tough. Try taking slow, deep breaths. Would you like some calming techniques?",
                "I'm here for you. What's been making you feel anxious? Let's talk it through.",
                "Feeling overwhelmed is hard. Would you like to try a grounding exercise or just chat?"
            ]
        },
        {
            "tag": "loneliness",
            "patterns": ["i feel lonely", "i'm alone", "no one cares", "i feel isolated", "i have no friends"],
            "responses": [
                "I'm so sorry you're feeling lonely. You're not alone here—I'm with you. Want to share more?",
                "Feeling isolated can be really hard. Would you like to talk about what's been going on?",
                "I care about you. Let's chat—maybe we can find something to lift your spirits."
            ]
        },
        {
            "tag": "anger",
            "patterns": ["i'm angry", "i feel mad", "i'm so frustrated", "i can't stop being irritated"],
            "responses": [
                "It's okay to feel angry sometimes. Want to tell me what's got you so frustrated?",
                "I'm here to listen. What's been making you feel this way?",
                "Anger can be tough to handle. Would you like some tips to cool down?"
            ]
        },
        {
            "tag": "coping_strategies",
            "patterns": ["how do i cope", "help me feel better", "what can i do to relax", "give me tips for stress", "how to deal with anxiety"],
            "responses": [
                "Here are a few ideas: try deep breathing (inhale for 4, hold for 4, exhale for 4), take a short walk, or write down your thoughts. Want more details on any of these?",
                "To relax, you could try listening to calming music or doing a quick mindfulness exercise. Would you like me to guide you through one?",
                "Coping with stress can start with small steps, like focusing on your breath or doing something you enjoy. Want to explore one of these?"
            ]
        },
        {
            "tag": "crisis",
            "patterns": ["i want to hurt myself", "i don't want to live", "i'm thinking of suicide", "i can't go on", "i feel hopeless"],
            "responses": [
                "I'm really worried about you, and I want you to know you're not alone. Please reach out to someone you trust or contact a helpline like 1800-599-0019. I'm here to listen to, can we talk more?",
                "Your life matters, and I'm here for you. Please consider calling a crisis hotline 1800-599-0019 or someone you trust. Can I help you find resources?",
                "I'm so sorry you're feeling this way. You're important, and help is available. Please call a helpline like 1800-599-0019 or talk to someone you trust. Want to share more with me?"
            ]
        },
        {
            "tag": "professional_help",
            "patterns": ["should i see a therapist", "do i need help", "should i talk to someone", "is therapy worth it"],
            "responses": [
                "Talking to a therapist can be really helpful for sorting through your feelings. It's a brave step! Would you like help finding resources to connect with one?",
                "If you're feeling overwhelmed, speaking with a professional can make a big difference. Want me to share some ways to find support?",
                "Therapy can offer new tools and perspectives. It's worth exploring if you feel stuck. Would you like tips on finding a therapist?"
            ]
        },
        {
            "tag": "positive",
            "patterns": ["i feel good", "i'm happy", "things are going well", "i'm excited"],
            "responses": [
                "That's so wonderful to hear! What's got you in such a great mood today?",
                "I love hearing that you're happy! Want to share what's making you smile?",
                "Yay, I'm thrilled you're feeling good! Tell me more about what's going well!"
            ]
        },
        {
            "tag": "confused",
            "patterns": ["i don't know what to do", "i'm confused", "i feel lost", "i don't understand myself"],
            "responses": [
                "Feeling lost can be really tough. Let's take it one step at a time—want to share what's confusing you?",
                "It's okay to feel unsure. I'm here to help you sort through it. What's on your mind?",
                "Confusion can feel overwhelming. Would you like to talk it out or try a small exercise to clear your thoughts?"
            ]
        },
        {
            "tag": "goodbye",
            "patterns": ["bye", "goodbye", "see you", "i'm leaving", "talk later"],
            "responses": [
                "Take care! I'm here whenever you need me. Come back soon!",
                "Goodbye for now! Feel free to reach out anytime you want to talk.",
                "See you later! I'm always here if you need support."
            ]
        },
        {
            "tag": "thanks",
            "patterns": ["thank you", "thanks", "appreciate it", "you're helpful"],
            "responses": [
                "You're so welcome! I'm glad I could help.",
                "Thanks for saying that! I'm here anytime you need me.",
                "No problem at all! Happy to support you."
            ]
        },
        {
            "tag": "no_response",
            "patterns": ["", " ", "..."],
            "responses": [
                "Not sure what to say? I'm here whenever you're ready to chat.",
                "Feeling quiet? That's okay! Let me know what's on your mind when you're ready.",
                "Just chilling? I'm here if you want to talk about anything."
            ]
        }
    ]

}

# Save intents to a JSON file
with open('intents.json', 'w') as file:
    json.dump(intents, file)

# Prepare training data
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes for later use
import pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build and train model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')


model = Sequential.from_config(model.get_config())
model.load_weights('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# NLP processing functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm not sure how to respond to that. Can you tell me more?"
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def store_chat(session_id, user_message, bot_response):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (session_id, user_message, bot_response) VALUES (?, ?, ?)",
              (session_id, user_message, bot_response))
    conn.commit()
    conn.close()


@app.route("/")
def home():
    init_db()
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    msg = request.form['msg']
    session_id = request.form.get('session_id', 'default_session')
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    store_chat(session_id, msg, res)
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)