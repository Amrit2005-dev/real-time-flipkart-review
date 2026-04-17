import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Set page config
st.set_page_config(page_title="Flipkart Review Sentiment", page_icon="🛍️", layout="centered")

# Ensure NLTK resources are available
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
        # this is basicall the web frame work is use for that

download_nltk_data()

# Clean input text function
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

# Load Models
@st.cache_resource
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
    vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    return None, None

model, vectorizer = load_models()

# Streamlit App UI
st.title("🛒 Flipkart Product Review Sentiment Analyzer")
st.markdown("""
This application analyzes real-time customer reviews for the **YONEX MAVIS 350 Nylon Shuttle**. 
Simply type or paste a review below to find out if it's considered **Positive** or **Negative**!
""")

st.write("---")

user_input = st.text_area("✍️ Enter your review here:", height=150, placeholder="E.g., The quality of the shuttle is amazing and it lasts very long...")

if st.button("Predict Sentiment 🚀"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model files not found. Please ensure the model has been trained and saved.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned_input = clean_text(user_input)
            
            if not cleaned_input.strip():
                st.warning("The input review doesn't contain enough valid words to analyze.")
            else:
                # Vectorize
                input_vector = vectorizer.transform([cleaned_input])
                
                # Predict
                prediction = model.predict(input_vector)[0]
                probabilities = model.predict_proba(input_vector)[0]
                
                confidence = max(probabilities) * 100
                
                st.write("---")
                st.subheader("Results:")
                
                if prediction == 1:
                    st.success(f"🌟 **Positive Review** (Confidence: {confidence:.2f}%)")
                    st.balloons()
                else:
                    st.error(f"😞 **Negative Review** (Confidence: {confidence:.2f}%)")
                    st.markdown("""
                    **Common Pain Points Often Associated with Negative Reviews:**
                    - "Fake / Duplicate Product"
                    - "High Price / Costly"
                    - "Breaks easily / Poor Durability"
                    - "Damaged Packaging"
                    """)
                    
st.caption("Developed using Streamlit, scikit-learn, and NLTK.")
