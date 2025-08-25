# Fake News Detection App 📰🚨

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## **Project Overview**

In today’s digital era, misinformation spreads rapidly across social media and online platforms. Fake news can influence public opinion, financial markets, and even health and safety decisions.  

This project implements a **Natural Language Processing (NLP) pipeline** using **Naive Bayes** to classify news articles as **Fake** or **True**. The app includes a **Streamlit interface** to allow users to input news text and receive real-time predictions.

---

## **Features**

- Detect whether a news article is **Fake ❌** or **True ✅**.  
- Interactive **Streamlit app** for real-time input.  
- Uses **title + text** for better accuracy.  
- Preprocessing with **lemmatization**, **stopword removal**, and **TF-IDF vectorization**.  
- Provides confidence scores for each prediction.  

---

## **Dataset**

The app uses two CSV files:

| File       | Description                       |
|-----------|-----------------------------------|
| True.csv  | Contains authentic news articles  |
| Fake.csv  | Contains fake news articles       |

Each file includes the columns:

- `title` – Headline of the news  
- `text` – Full article content  
- `subject` – Category/topic of the news  
- `date` – Publication date  

The datasets are combined and labeled:

- `1` → True news  
- `0` → Fake news  

---

## **Project Workflow**

1. **Data Loading & Labeling**  
   - Load CSVs using Pandas.  
   - Add labels for True/Fake news.  
   - Combine datasets and shuffle.  

2. **Data Preprocessing**  
   - Lowercasing, removing punctuation, numbers, HTML tags, extra whitespace.  
   - Tokenization and stopword removal.  
   - Lemmatization for word normalization.  
   - Cleaned text stored in `clean_content`.

3. **Exploratory Data Analysis (EDA)**  
   - Visualize label distribution.  
   - Word count distribution per class.  
   - Top 20 words per class.  
   - Word clouds for Fake vs True news.

4. **Feature Extraction**  
   - TF-IDF vectorization (max_features=5000).  
   - Optional: n-grams for better context.  

5. **Model Building**  
   - **Multinomial Naive Bayes** trained on TF-IDF features.  
   - Evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.  

6. **Streamlit App**  
   - Input news text and predict Fake/True in real-time.  
   - Display prediction and confidence score.  

---

## **Screenshots**



### **True News Prediction**  
![True News Prediction](<screenshots/true.png.png>
)

### **Fake News Prediction**  
![Fake News Prediction](<screenshots/fake.png.png>
)



---

## **Author**

**Om Raut**  
- GitHub: [https://github.com/omraut31](https://github.com/omraut31)  
- Medium: [https://omraut31.medium.com](https://omraut31.medium.com)  
- LinkedIn: [https://www.linkedin.com/in/omraut31](https://www.linkedin.com/in/omraut31)
