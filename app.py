from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import os

# Initialize the Flask application
app = Flask(__name__)

# Paths for saving model and vectorizer
model_section_filename = 'section_prediction_model_svm.pkl'
vectorizer_filename = 'tfidf_vectorizer_svm.pkl'
data_file_path = 'ipc_sections.csv'  # Path to your dataset

def train_model():
    # Load your dataset
    data = pd.read_csv(data_file_path)

    # Fill missing values in 'Description' with an empty string
    data['Description'] = data['Description'].fillna('')

    # Prepare the target variable
    y_section = data['Section']

    # Preprocess and Vectorize the text data using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(data['Description']).toarray()

    # Initialize and train a Support Vector Machine model for Section prediction
    model_section = SVC(kernel='linear', random_state=42)
    model_section.fit(X, y_section)

    # Save the trained model and TF-IDF vectorizer to files
    joblib.dump(model_section, model_section_filename)
    joblib.dump(tfidf_vectorizer, vectorizer_filename)

    print("Model and vectorizer saved successfully.")

# Train the model when the application starts
train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    description = request.form['description']
    
    # Load the trained model and vectorizer
    loaded_model_section = joblib.load(model_section_filename)
    loaded_vectorizer = joblib.load(vectorizer_filename)
    
    # Vectorize the input description
    description_vectorized = loaded_vectorizer.transform([description]).toarray()
    
    # Make predictions
    predicted_section = loaded_model_section.predict(description_vectorized)[0]

    # Load the original data to find corresponding punishment
    data = pd.read_csv(data_file_path)  # Ensure this is the same dataset used for training
    predicted_punishment = data[data['Section'] == predicted_section]['Punishment'].values
    
    if len(predicted_punishment) > 0:
        punishment = predicted_punishment[0]
    else:
        punishment = "Not found"
    
    return jsonify({
        'predicted_section': predicted_section,
        'predicted_punishment': punishment
    })

if __name__ == '__main__':
    app.run(debug=True)
