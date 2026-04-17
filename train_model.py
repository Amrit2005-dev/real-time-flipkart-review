import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import joblib
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Set fallback for NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def train_and_log():
    # MLflow Experiment Setup
    mlflow.set_experiment("Flipkart_Sentiment_Analysis")
    
    with mlflow.start_run(run_name="Logistic_Regression_Standard"):
        # Log basic project tags
        mlflow.set_tag("project", "Flipkart Review Sentiment")
        mlflow.set_tag("model_type", "Logistic Regression")

        print("Loading and Preprocessing data...")
        df = pd.read_csv(r"C:\flipkart\reviews_data_dump\reviews_badminton\data.csv")
        df = df.dropna(subset=['Review text', 'Ratings'])
        df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)
        df['Clean_Text'] = df['Review text'].apply(clean_text)

        # Embedding 
        print("Embedding text...")
        max_features = 5000
        ngram_range = (1, 2)
        
        # Log manual parameters (autolog will catch some, but being explicit is good)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", str(ngram_range))

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X = vectorizer.fit_transform(df['Clean_Text'])
        y = df['Sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Enable Autologging for sklearn
        mlflow.sklearn.autolog()
        
        print("Training model...")
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predictions and Metrics
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average='weighted')
        mlflow.log_metric("f1_score", f1)
        print(f"Model F1-Score: {f1:.4f}")

        # --- ARTIFACTS: PLOTS ---
        print("Generating plots...")
        # 1. Confusion Matrix
        fig_cm, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=["Negative", "Positive"], ax=ax, cmap='Blues')
        plt.title("Confusion Matrix")
        mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")
        plt.close(fig_cm)

        # 2. Coefficients Plot (Top Feature Importance)
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        top_positive_idx = np.argsort(coefficients)[-15:]
        top_negative_idx = np.argsort(coefficients)[:15]
        
        top_coeffs = np.concatenate([coefficients[top_negative_idx], coefficients[top_positive_idx]])
        top_features = np.concatenate([feature_names[top_negative_idx], feature_names[top_positive_idx]])
        
        fig_coeff = plt.figure(figsize=(12, 8))
        colors = ['red' if c < 0 else 'blue' for c in top_coeffs]
        plt.barh(top_features, top_coeffs, color=colors)
        plt.title("Top 15 Positive and Negative Word Coefficients")
        plt.xlabel("Coefficient Value")
        mlflow.log_figure(fig_coeff, "plots/feature_coefficients.png")
        plt.close(fig_coeff)

        # Save local artifacts
        joblib.dump(model, "best_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        
        # Register the Model in Registry
        print("Registering model in MLflow Registry...")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "Flipkart_Sentiment_LR")
        
        print("Training run complete.")

if __name__ == "__main__":
    train_and_log()
