# Quick Start Guide

## Install & Run in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

## What Happens Next?

1. Browser opens at `http://localhost:8501`
2. Upload your PDF bank statement
3. Click "Analyze Statement"
4. Wait 20-30 seconds for AI processing
5. View comprehensive insights and charts!

## First Time Users

- Use the test statement included (`test_statement.pdf`) to try it out
- Processing takes ~25 seconds for 100 transactions
- All data stays local - nothing is stored permanently

## Need Help?

Check [README.md](README.md) for detailed documentation and troubleshooting.
