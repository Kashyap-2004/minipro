import os
import sys
sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv
load_dotenv()

# Test the predict_news function
from fake_news_pipeline import predict_news

test_news = "Donald Trump wins the 2024 presidential election with record-breaking votes."
print(f"Testing with news: {test_news}\n")

try:
    result, confidence = predict_news(test_news)
    print(f"✅ SUCCESS!")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence}%")
except Exception as e:
    print(f"❌ FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

