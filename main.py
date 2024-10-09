# ... existing imports ...
from flask import Flask, request, jsonify
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Load the model
model = joblib.load('toxic_comment_classifier.pkl')

app = Flask(__name__)

# Initialize NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def stemming(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

@app.route('/check_comment', methods=['POST'])
def check_comment():
    data = request.json
    comment = data.get('comment', '')

    # Preprocess the comment
    cleaned_comment = clean_text(comment)
    cleaned_comment = remove_stopwords(cleaned_comment)
    cleaned_comment = stemming(cleaned_comment)

    # Debugging: Print the cleaned comment
    print(f"Cleaned Comment: {cleaned_comment}")

    # Predict toxicity
    prediction = model.predict([cleaned_comment])  # Get the prediction

    # Debugging: Print the prediction result
    print(f"Prediction: {prediction}")

    # Check if the prediction indicates toxicity
    if isinstance(prediction, (list, np.ndarray)):
        if any(prediction):  # Check if any label is predicted as toxic
            print("Comment is toxic.")
            return jsonify({'message': 'Comment is toxic.'}), 200
        else:
            print("Comment is not toxic.")
            return jsonify({'message': 'Comment is not toxic.'}), 200
    else:
        # If prediction is a single value
        if prediction == 1:  # Assuming 1 indicates toxic
            print("Comment is toxic.")
            return jsonify({'message': 'Comment is toxic.'}), 200
        else:
            print("Comment is not toxic.")
            return jsonify({'message': 'Comment is not toxic.'}), 200

if __name__ == '__main__':
    app.run(debug=True)