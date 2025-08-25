# fake_news_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>','', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# Load datasets
@st.cache_data
def load_data():
    true_df = pd.read_csv("dataset/True.csv")
    fake_df = pd.read_csv("dataset/Fake.csv")
    true_df["label"] = 1
    fake_df["label"] = 0
    data = pd.concat([true_df, fake_df], axis=0).sample(frac=1).reset_index(drop=True)
    data["clean_content"] = data["text"].apply(clean_text)
    return data

data = load_data()

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data["clean_content"])
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("""
This app predicts whether a news article is **Fake** or **True** using NLP and Naive Bayes.
""")

# Text input
user_input = st.text_area("Enter the news article here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        # Preprocess and vectorize input
        cleaned_input = clean_text(user_input)
        input_vec = tfidf.transform([cleaned_input])
        prediction = nb_model.predict(input_vec)[0]
        probability = nb_model.predict_proba(input_vec)[0]

        if prediction == 1:
            st.success(f"Prediction: **True News** ✅\nConfidence: {probability[1]*100:.2f}%")
        else:
            st.error(f"Prediction: **Fake News** ❌\nConfidence: {probability[0]*100:.2f}%")
