import json
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()


with open('admission.json', 'r') as file:
    intents = json.load(file)['intents']

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

patterns = []
tags = []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(preprocess_text(pattern))
        tags.append(intent['tag'])


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)


def classify_input(user_input):
    user_input = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vec, X)
    max_similarity_index = similarities.argmax()
    max_similarity_score = similarities[0, max_similarity_index]
    if max_similarity_score > 0.2:  # Set a threshold for similarity
        return tags[max_similarity_index]
    return "noanswer"

def get_response(tag):
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."

def save_intents(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def chatbot():
    print("Chatbot: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        tag = classify_input(user_input)
        if tag == "noanswer":
            print("Chatbot: I don't know the answer. Can you teach me?")
            new_response = input('Type the answer or "skip" to skip: ')
            if new_response.lower() != 'skip':
                # Find the index of the intent with the "noanswer" tag
                index = next((i for i, intent in enumerate(intents) if intent["tag"] == "noanswer"), None)
                if index is not None:
                    
                    intents[index]["responses"].append(new_response)
                    print("Chatbot: Thank you! I learned a new response.")
                    save_intents('intents.json', {"intents": intents})  
                else:
                    print("Chatbot: Error - Intent not found.")
        else:
            response = get_response(tag)
            print(f"Chatbot: {response}")


if __name__ == "__main__":
    # Initialize the vectorizer and fit it on patterns
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)
    chatbot()
