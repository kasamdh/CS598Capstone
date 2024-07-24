import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Load the trained Multinomial Naive Bayes model and the CountVectorizer object
model_path = 'model.pkl'
vectorizer_path = 'cv-model.pkl'

with open(model_path, 'rb') as model_file, open(vectorizer_path, 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('input_form.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        user_input = request.form['text']
        input_data = [user_input]
        vectorized_input = vectorizer.transform(input_data).toarray()
        prediction = model.predict(vectorized_input)
        return render_template('input_form.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)



