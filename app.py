import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import spacy
import re
from sklearn.base import BaseEstimator, ClassifierMixin

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class MultinomialNaiveBayes(BaseEstimator, ClassifierMixin):
    #function to initialize
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    #function fit to calulate the prior and likelihood
    def fit(self, X_train, Y_train):
        noOfDocs, noOfWords = X_train.shape
        self.classes_ = np.unique(Y_train)
        noOfClasses = len(self.classes_)
        
        self.class_prior = np.zeros(noOfClasses)
        for index, cls in enumerate(self.classes_):
            self.class_prior[index] = np.sum(Y_train == cls) / noOfDocs
        
        self.conditional_prob = np.zeros((noOfClasses, noOfWords))
        for index, cls in enumerate(self.classes_):
            xTrain_cls = X_train[Y_train == cls]
            class_word_count = xTrain_cls.sum(axis=0)
            total_class_word_count = class_word_count.sum()
            self.conditional_prob[index, :] = (class_word_count + self.alpha) / (total_class_word_count + noOfWords)  # Laplace smoothing
    
    #function predict to test the trained data     
    def predict(self, X_train):
        noOfDocs = X_train.shape[0]
        probabilityLog = np.zeros((noOfDocs, len(self.classes_)))
        
        for index, cls in enumerate(self.classes_):
            probabilityLogCls = np.log(self.class_prior[index])
            probabilityLogWords = X_train @ np.log(self.conditional_prob[index, :].T)
            probabilityLog[:, index] = probabilityLogCls + probabilityLogWords
        
        return self.classes_[np.argmax(probabilityLog, axis=1)]
    
    #function to evaluate the predicted data using ROC and AUC
    def predict_proba(self, X_train):
        noOfDocs = X_train.shape[0]
        probabilityLog = np.zeros((noOfDocs, len(self.classes_)))
        
        for index, cls in enumerate(self.classes_):
            probabilityLogCls = np.log(self.class_prior[index])
            probabilityLogWords = X_train @ np.log(self.conditional_prob[index, :].T)
            probabilityLog[:, index] = probabilityLogCls + probabilityLogWords
        
        return np.exp(probabilityLog - np.max(probabilityLog, axis=1, keepdims=True))
    

# Load the saved model
model_filename = 'model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the saved TfidfVectorizer
vectorizer_filename = 'vectorizer.pkl'
with open(vectorizer_filename, 'rb') as file:
    vectorizer = pickle.load(file)

# Function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text.lower())
    text = text.replace("n't", " not")
    return ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    
    text = request.form['text']
    preprocessed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text])

    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0][prediction]

    sentiment_map = {1: 'Positive', 0: 'Negative'}
    sentiment = sentiment_map.get(prediction, 'Neutral')

    sentiment_class = 'positive' if sentiment == 'Positive' else 'negative' if sentiment == 'Negative' else 'neutral'

    return render_template('result.html', sentiment=sentiment, probability=f"{probability:.2f}", sentiment_class=sentiment_class)


if __name__ == '__main__':
    app.run(debug=True)

