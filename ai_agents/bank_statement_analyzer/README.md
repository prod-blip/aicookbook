# 💰 Bank Statement Analyzer

An AI-powered bank statement analyzer that automatically extracts transactions from PDF statements, categorizes spending using GPT-4o, and provides comprehensive financial insights with visual analytics.



https://github.com/user-attachments/assets/10f5e49a-a54a-4da7-afa9-58f097ced095



## ✨ Features

- **Automatic Transaction Extraction** - Upload PDF bank statements and get all transactions extracted automatically using AI
- **Smart Categorization** - GPT-4o intelligently categorizes each transaction into 9 predefined categories (Food, Transport, Shopping, etc.)
- **Recurring Expense Detection** - Automatically identifies recurring payments like subscriptions, bills, and regular purchases
- **Visual Analytics** - Interactive charts showing category breakdown, daily spending trends, and top merchants
- **Downloadable Reports** - Export complete analysis as text reports for record-keeping

## 🚀 Setup

### Requirements

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
```bash
cd ai_agents/bank_statement_analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Your `.env` file should contain:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 🎯 Running the Application

### Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

This will open a web interface at `http://localhost:8501` where you can:
1. Upload your PDF bank statement
2. Click "Analyze Statement"
3. View comprehensive insights and charts
4. Download analysis report

### Command Line (For Testing)

```bash
python agent.py
```

Note: You'll need to modify the `test_pdf_path` in `agent.py` to point to your statement.

## 📊 Analysis Output

The analyzer provides the following insights:

### 1. Summary Metrics
- Total Spent (Debit)
- Total Received (Credit)
- Net Balance Change
- Top Spending Category
- Number of Recurring Expenses

### 2. Category Breakdown
Transactions are automatically categorized into:
- **Food & Dining** (restaurants, cafes, food delivery, groceries)
- **Transport** (fuel, Uber, Rapido, metro, parking, tolls)
- **Shopping** (clothing, electronics, retail, e-commerce)
- **Utilities** (electricity, water, gas, internet, phone bills)
- **Bills & Recharges** (mobile recharge, subscriptions, insurance)
- **Entertainment** (movies, concerts, gaming, streaming)
- **Healthcare** (pharmacy, doctor, hospital, medical)
- **Transfers** (bank transfers, loan payments, credit card payments)
- **Miscellaneous Payments** (person-to-person UPI transfers, payments to individuals)
- **Others** (anything that doesn't fit above)

### 3. Visual Insights
- **Pie Chart**: Category-wise spending distribution
- **Line Chart**: Daily spending trend over the statement period
- **Bar Chart**: Top 10 merchants by total spending

### 4. Recurring Expenses
Automatically detects and lists:
- Merchant name and category
- Frequency (number of occurrences)
- Average amount per transaction
- Total spent on that merchant

### 5. Transaction Table
Complete list of all transactions with:
- Date
- Description
- Amount
- Type (Debit/Credit)
- Category
- Filter by category option

## 🏗️ Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BANK STATEMENT ANALYZER                   │
│                     (LangGraph Workflow)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 1: load_pdf                                           │
│  - Extracts text from PDF using PyPDF                       │
│  - Output: raw_text, page_count                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 2: parse_transactions                                 │
│  - LLM converts text to structured JSON transactions        │
│  - Extracts: date, description, amount, type                │
│  - Output: transactions[], metadata (account, period)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 3: categorize_transactions                            │
│  - GPT-4o assigns category to each transaction              │
│  - Batch processing (all txns in one call)                  │
│  - Output: categorized_transactions[]                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 4: analyze_spending                                   │
│  - Calculate category totals                                │
│  - Detect recurring expenses (2+ occurrences)               │
│  - Compute daily trends and top merchants                   │
│  - Output: analytics (totals, recurring, trends)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Node 5: prepare_visualizations                             │
│  - Structure data for charts                                │
│  - Output: chart_data (pie, line, bar)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                          ┌───────┐
                          │  END  │
                          └───────┘
```

**Key Design Decisions:**
- **Sequential Pipeline**: All 5 nodes execute in order
- **LLM-based Parsing**: Handles various PDF formats robustly
- **Batch Categorization**: Single API call for all transactions (efficient)
- **Pure Python Analytics**: Fast calculations without additional LLM calls
- **State Preservation**: `{**state, "field": value}` pattern maintains all data

## ⚠️ Important Notes

### Security & Privacy
- **Local Processing**: All analysis happens on your machine
- **No Data Storage**: Uploaded files are temporarily stored in `/tmp/` and not persisted
- **API Usage**: Only statement text is sent to OpenAI for parsing/categorization
- **Recommendation**: Do not upload statements from production/live accounts in shared environments

### API Costs
- **Model**: GPT-4o
- **Average Cost per Statement**: ~$0.03-0.05 (100-200 transactions)
- **Breakdown**:
  - Transaction parsing: 1 API call (~$0.01)
  - Categorization: 1 API call (~$0.02)
  - Total: 2 API calls per statement

### Performance
- **Processing Time**: 20-30 seconds for typical statement (100 transactions)
- **Breakdown**:
  - PDF Load: <2s
  - Transaction Parsing: 5-8s
  - Categorization: 10-15s
  - Analysis: <1s
  - Visualization: <1s

### Limitations
- **PDF Format**: Works best with digital PDFs (not scanned/image-based)
- **Statement Layout**: Optimized for Indian bank statements; may need prompt tuning for other formats
- **Transaction Volume**: Tested with up to 500 transactions; larger statements may hit token limits
- **Date Format**: Assumes DD-MM-YYYY (Indian format); can be adjusted in prompts

## 📚 Tech Stack

- **Framework**: LangGraph (agentic workflow orchestration)
- **LLM**: OpenAI GPT-4o (parsing & categorization)
- **PDF Processing**: PyPDF (text extraction)
- **Frontend**: Streamlit (web interface)
- **Visualization**: Plotly (interactive charts)
- **Data Analysis**: Pandas (transaction tables)

## 🔗 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Documentation](https://plotly.com/python/)

## 📝 License

This project is part of the AI Cookbook repository. Feel free to use and modify for personal or commercial use.

---

**Built with ❤️ for smarter financial insights**

Have questions or suggestions? Open an issue or contribute to the project!
