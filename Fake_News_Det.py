from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfvect = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
loaded_model = pickle.load(open('model1.pkl', 'rb'))
dataframe = pd.read_csv('cleaned_IFND.csv')
x = dataframe['Statement']
y = dataframe['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

import numpy as np

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)

    # Get confidence from decision function
    decision_score = loaded_model.decision_function(vectorized_input_data)
    
    # Convert to pseudo-confidence between 0–100%
    confidence = 1 / (1 + np.exp(-abs(decision_score[0]))) * 100  # sigmoid scaling
    
    return prediction, round(confidence, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred, confidence = fake_news_det(message)
        print(pred, confidence)
        return render_template('index.html', prediction=pred, confidence=confidence, message=message)
    else:
        return render_template('index.html', prediction="Something went wrong", confidence=None, message="")

if __name__ == '__main__':
    app.run(debug=True)