import os
import pandas as pd
import re
import nltk
import google.generativeai as genai

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Lazy initialization of Gemini API
_gemini_initialized = False

def _init_gemini():
    global _gemini_initialized
    if not _gemini_initialized:
        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file")
        genai.configure(api_key=API_KEY)
        _gemini_initialized = True

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

def load_and_prepare_data():
    data_frames = []

    # Load True.csv
    if os.path.exists("True.csv"):
        true_df = pd.read_csv("True.csv")
        true_df['label'] = 1
        data_frames.append(true_df)
    else:
        print("⚠️ True.csv not found.")

    # Load Fake.csv
    if os.path.exists("Fake.csv"):
        fake_df = pd.read_csv("Fake.csv")
        fake_df['label'] = 0
        data_frames.append(fake_df)
    else:
        print("⚠️ Fake.csv not found.")

    # Load news_dataset.csv
    feedback_path = "news_dataset.csv"
    if os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0:
        feedback_df = pd.read_csv(feedback_path)
        if 'text' in feedback_df.columns and 'label' in feedback_df.columns:
            feedback_df = feedback_df[['text', 'label']]
            data_frames.append(feedback_df)
        else:
            print("⚠️ news_dataset.csv missing 'text' or 'label' columns.")
    else:
        print("⚠️ news_dataset.csv is empty or missing.")

    if not data_frames:
        raise ValueError("❌ No valid datasets found. Cannot train.")

    # Combine all
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Merge title+text if both present (from True/Fake)
    if 'title' in combined_df.columns and 'text' in combined_df.columns:
        combined_df['text'] = combined_df['title'].astype(str) + " " + combined_df['text'].astype(str)

    if 'text' not in combined_df.columns or 'label' not in combined_df.columns:
        raise ValueError("❌ Required columns ('text', 'label') missing after merge.")

    # Clean text
    combined_df['text'] = combined_df['text'].astype(str).apply(clean_text)

    # Shuffle and return
    combined_df = combined_df[['text', 'label']].sample(frac=1).reset_index(drop=True)
    return combined_df['text'], combined_df['label']

def train_model(X, y, show_metrics=True):
    if len(set(y)) < 2:
        print("❌ Cannot train model: only one class present in data:", set(y))
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=1.0,        # Safer for small datasets
        min_df=1,          # Keep as 1
        ngram_range=(1, 1) # Simple for dynamic data
    )

    X_tfidf = vectorizer.fit_transform(X)

    model = LogisticRegression(C=1.5, max_iter=1000, solver='liblinear')
    model.fit(X_tfidf, y)

    if show_metrics:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        if len(set(y_train)) < 2:
            print("⚠️ Skipping evaluation: train set only has one class.")
        else:
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)

            print("✅ Model Retrained on Approved Data")
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
            print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    dump(model, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)
    print("✅ Model and vectorizer saved.")

    return model, vectorizer

def load_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = load(MODEL_PATH)
        vectorizer = load(VECTORIZER_PATH)
        print("✅ Model and vectorizer loaded from disk.")
        return model, vectorizer
    else:
        print("⚠️ Model files not found. Please train the model first.")
        return None, None

def predict_news_gemini(news):
    """
    Use Gemini API to predict if news is real or fake.
    """
    import json
    
    try:
        print("[GEMINI] Initializing API...")
        # Initialize Gemini on first use
        _init_gemini()
        print("[GEMINI] API initialized")
        
        print("[GEMINI] Creating model...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("[GEMINI] Model created")
        
        prompt = f"""Analyze the following news text and determine if it's REAL or FAKE news.
Return ONLY valid JSON (no markdown, no code blocks, just raw JSON):
{{"prediction": "Real" or "Fake", "confidence": 0-100}}

News: {news}"""
        
        print("[GEMINI] Sending request to API...")
        response = model.generate_content(prompt)
        print("[GEMINI] Received response from API")
        
        # Handle response - it might be response.text or str(response)
        if hasattr(response, 'text'):
            response_text = response.text.strip()
        else:
            response_text = str(response).strip()
        
        print(f"[GEMINI] Raw API response: {response_text[:200]}")
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            prediction = str(result.get('prediction', 'Fake'))
            confidence = int(result.get('confidence', 50))
            print(f"[GEMINI] Parsed successfully: prediction={prediction}, confidence={confidence}")
            return prediction, confidence
        except json.JSONDecodeError as je:
            print(f"[GEMINI] JSON Parse Error: {je}")
            print(f"[GEMINI] Response was: {response_text}")
            # Fallback parsing
            if 'real' in response_text.lower():
                return "Real", 70
            else:
                return "Fake", 70
            
    except Exception as e:
        print(f"[GEMINI] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Unable to determine", 0

def predict_news(news, model=None, vectorizer=None, threshold=0.55):
    """
    Main prediction function - uses Gemini API.
    Legacy parameters (model, vectorizer) kept for backward compatibility.
    """
    return predict_news_gemini(news)

def predict_image_gemini(image_path):
    """
    Use Gemini API to analyze an image and determine if it contains fake news.
    """
    import json
    import base64
    
    try:
        print("[GEMINI-IMAGE] Initializing API...")
        _init_gemini()
        print("[GEMINI-IMAGE] API initialized")
        
        # Read and encode the image
        with open(image_path, 'rb') as img_file:
            image_data = base64.standard_b64encode(img_file.read()).decode('utf-8')
        
        print("[GEMINI-IMAGE] Sending image to API...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """Analyze this image carefully and determine if it appears to contain fake news, misleading information, or misinformation.
Consider:
- Text content visible in the image
- Visual elements and their authenticity
- Common fake news indicators (manipulated images, misleading captions, etc.)

Return ONLY valid JSON (no markdown, no code blocks, just raw JSON):
{"prediction": "Real" or "Fake", "confidence": 0-100, "reason": "brief explanation"}"""
        
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_data}])
        print("[GEMINI-IMAGE] Received response from API")
        
        if hasattr(response, 'text'):
            response_text = response.text.strip()
        else:
            response_text = str(response).strip()
        
        print(f"[GEMINI-IMAGE] Raw API response: {response_text[:200]}")
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        try:
            result = json.loads(response_text)
            prediction = str(result.get('prediction', 'Fake'))
            confidence = int(result.get('confidence', 50))
            print(f"[GEMINI-IMAGE] Parsed successfully: prediction={prediction}, confidence={confidence}")
            return prediction, confidence
        except json.JSONDecodeError as je:
            print(f"[GEMINI-IMAGE] JSON Parse Error: {je}")
            print(f"[GEMINI-IMAGE] Response was: {response_text}")
            if 'real' in response_text.lower():
                return "Real", 70
            else:
                return "Fake", 70
            
    except Exception as e:
        print(f"[GEMINI-IMAGE] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Unable to determine", 0

def main():
    print("\n📦 Loading and preparing data...")
    X, y = load_and_prepare_data()

    print("\n🔧 Training model...")
    model, vectorizer = train_model(X, y)

    if not model or not vectorizer:
        print("❌ Model training failed.")
        return

    while True:
        user_input = input("\n📰 Enter news text (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("👋 Exiting.")
            break

        label, confidence = predict_news(user_input, model, vectorizer)
        print(f"✅ Prediction: The news is **{label}** with {confidence}% confidence.")

if __name__ == "__main__":
    main()
