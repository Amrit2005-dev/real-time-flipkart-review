import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def main():
    # Ensure nltk resources are downloaded
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    print("Loading data...")
    df = pd.read_csv(r"C:\flipkart\reviews_data_dump\reviews_badminton\data.csv")
    
    print(f"Data Shape: {df.shape}")
    
    # Handle missing values
    df = df.dropna(subset=['Review text', 'Ratings'])
    
    # Create Labels: Ratings > 3 -> Positive (1), else Negative (0)
    df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)
    
    print("Class distribution:")
    print(df['Sentiment'].value_counts())
    
    print("Cleaning text...")
    df['Clean_Text'] = df['Review text'].apply(clean_text)
    
    # Identify pain points
    negative_reviews = df[df['Sentiment'] == 0]['Clean_Text']
    words = ' '.join(negative_reviews).split()
    word_counts = pd.Series(words).value_counts().head(20)
    print("\nTop words in negative reviews (Pain Points):")
    print(word_counts)
    
    # Embedding 
    print("\nApplying TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['Clean_Text'])
    y = df['Sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced')
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    print("\nLogistic Regression F1-Score:")
    print(classification_report(y_test, lr_preds))
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print("\nRandom Forest F1-Score:")
    print(classification_report(y_test, rf_preds))
    
    print("Saving model and vectorizer...")
    joblib.dump(lr_model, "best_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Training complete and models saved successfully.")

if __name__ == "__main__":
    main()
