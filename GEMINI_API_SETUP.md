# Gemini API Integration Setup

## What Was Changed

Your fake news detector has been updated to use **Google Gemini API** instead of the local ML model. This is lighter, faster, and API-based as requested.

### Changes Made:

1. **fake_news_pipeline.py**
   - Added `google.generativeai` import
   - Added `load_dotenv()` for environment variables
   - Created new `predict_news_gemini()` function that uses Gemini API
   - Updated `predict_news()` to call Gemini API instead of local model
   - Legacy model parameters kept for backward compatibility

2. **Home.py**
   - Removed model training logic (no longer needed)
   - Updated prediction call to use only `predict_news(news_text)` 
   - Kept MongoDB and email functionality intact

3. **.env file**
   - Added `GEMINI_API_KEY` variable

## Setup Instructions

### 1. Get Your Gemini API Key
- Go to [Google AI Studio](https://aistudio.google.com/apikey)
- Create a new API key
- Copy it

### 2. Update .env File
Replace `your_gemini_api_key_here` with your actual Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Install Dependencies
```powershell
pip install google-generativeai
```

### 4. Run the Application
```powershell
python Home.py
```

## How It Works

When you submit news text:
1. Flask sends it to `predict_news()` function
2. Function calls Gemini API with a prompt asking to classify the news
3. Gemini returns: prediction (Real/Fake), confidence score, and reason
4. Result is displayed on the webpage
5. User feedback is still stored in MongoDB for future improvements

## Benefits

✅ **No local model needed** - lighter memory footprint
✅ **API-based** - always up-to-date with latest Gemini model
✅ **Faster** - no training overhead
✅ **Scalable** - Google handles the infrastructure
✅ **Better accuracy** - Gemini's advanced language understanding

## Troubleshooting

If you get "GEMINI_API_KEY not found":
- Check your `.env` file has the correct key
- Make sure you're in the right directory when running the app
- Verify the `.env` file is in the same folder as `Home.py`

If you get API errors:
- Check your internet connection
- Verify your API key is valid
- Check your Google Cloud quota (free tier has limits)
