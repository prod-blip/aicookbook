# 📊 Data Analysis Agent - Natural Language Data Analysis

An intelligent data analysis agent that understands natural language questions and automatically generates insights, statistics, and visualizations from your CSV/Excel files.

✨ **Powered by LangGraph + GPT-4o for conversational data exploration**


https://github.com/user-attachments/assets/80276852-f65e-4969-820a-11ae8c9e3bdc


## Features

* 🤖 **Natural Language Queries** - Ask questions in plain English, no coding required
* 📊 **Automatic Visualizations** - Intelligently generates charts when they enhance understanding
* 💡 **AI-Powered Insights** - GPT-4o analyzes your data and provides clear, actionable insights
* 📁 **Multiple Formats** - Supports CSV and Excel files with automatic encoding detection
* 🔒 **Secure & Local** - Your data stays on your machine, only queries sent to OpenAI
* 📥 **Download Results** - Export analysis reports and charts for documentation
* 📜 **Analysis History** - Track multiple queries in a single session
* ⚡ **Fast Processing** - Efficient 6-node LangGraph architecture for quick results

## Setup

### Requirements

* Python 3.8+
* OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))
* Modern web browser for Streamlit UI

### Installation

1. Clone this repository:
```bash
git clone https://github.com/prod-blip/aicookbook.git
cd aicookbook/ai_agents/data_analysis_agent
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Get your OpenAI API credentials:
   * Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   * Create a new API key
   * Copy the key for next step

4. Setup your `.env` file:

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Or** enter the API key directly in the Streamlit sidebar when you run the app.

## Running the App

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Your browser will automatically open to `http://localhost:8501`

3. Follow these steps in the UI:
   * Enter your OpenAI API key in the sidebar (if not in .env)
   * Upload a CSV or Excel file
   * Type your question in natural language
   * Click "🚀 Analyze Data"
   * View insights and download results

## Analysis Capabilities

### What You Can Ask

**Statistical Analysis:**
- "What is the average sales by region?"
- "Show me the median income across categories"
- "Calculate total revenue per product"

**Trend Analysis:**
- "What's the sales trend over time?"
- "How did performance change month by month?"
- "Show me growth patterns"

**Comparative Analysis:**
- "Which category has the highest revenue?"
- "Compare performance across different regions"
- "What are the top 5 products by sales?"

**Pattern Detection:**
- "Find outliers in the price data"
- "Are there any anomalies?"
- "What patterns exist in customer behavior?"

**Distribution Analysis:**
- "Show the distribution of ages"
- "What's the frequency of each category?"
- "How are values spread across groups?"

### Output Format

Each analysis provides:

1. **Natural Language Response** - Clear, conversational answer with emojis
2. **Key Statistics** - Relevant numbers and metrics
3. **Visual Chart** (when helpful) - Automatically generated matplotlib visualization
4. **Downloadable Results** - Text report and PNG chart

## Agent Architecture

The application uses **LangGraph** with a **sequential workflow pattern** consisting of 6 specialized nodes:
```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Data Analyst Agent                        │
└─────────────────────────────────────────────────────────────────┘

         User uploads file + asks question
                      ↓
         ┌──────────────────────────┐
         │   1. LOAD DATA NODE      │
         │  ─────────────────────   │
         │  • Read CSV/Excel        │ │
         └──────────────────────────┘
                      ↓
         ┌──────────────────────────┐
         │  2. UNDERSTAND QUERY     │
         │  ─────────────────────   │
         │  • Analyze user question │
         │  • Inspect data structure││
         └──────────────────────────┘
                      ↓
         ┌──────────────────────────┐
         │  3. GENERATE CODE        │
         │  ─────────────────────   │
         │  • LLM writes Python code│   │
         └──────────────────────────┘
                      ↓
         ┌──────────────────────────┐
         │  4. EXECUTE ANALYSIS     │
         │  ─────────────────────   │
         │  • Run generated code    │     │
         └──────────────────────────┘
                      ↓
         ┌──────────────────────────┐
         │  5. GENERATE CHART       │
         │  ─────────────────────   │
         │  • LLM creates viz code  │   │
         │  • Store image           │
         └──────────────────────────┘
                      ↓
         ┌──────────────────────────┐
         │  6. CREATE RESPONSE      │
         │  ─────────────────────   │
         │  • LLM formats answer    │
         │  • Add insights & context│
         │  • Include statistics    │
         │  • Return to user        │
         └──────────────────────────┘
                      ↓
         Display results + chart in Streamlit UI
```

### Node Details

1. **load_data_node**: Loads CSV/Excel files with multiple encoding fallbacks, validates format, stores DataFrame in state
2. **understand_query_node**: Uses GPT-4o to interpret user's question, creates analysis plan, determines visualization needs
3. **generate_code_node**: LLM generates executable Python code for analysis, handles numeric vs categorical columns intelligently
4. **execute_analysis_node**: Safely executes generated code in controlled environment, stores results with type conversion
5. **generate_chart_node**: Creates matplotlib visualizations when beneficial, converts to base64 for display
6. **create_response_node**: Formats final answer in natural language with insights, statistics, and context

## Important Notes

⚠️ **Data Privacy**: Your uploaded files are processed locally. Only your questions and data summaries are sent to OpenAI's API for analysis generation.

🔐 **API Key Security**: Never commit your `.env` file to version control. The `.gitignore` is configured to exclude it.

💰 **API Costs**: Each analysis uses GPT-4o (3-4 API calls per query). Monitor your OpenAI usage at [platform.openai.com](https://platform.openai.com/usage).

📊 **File Size Limits**: Streamlit has a default 200MB upload limit. For larger files, increase with `server.maxUploadSize` in `.streamlit/config.toml`.

⚡ **Performance**: First query may take 10-15 seconds as the LLM generates code. Subsequent queries on same data are faster.

🎯 **Best Results**: Be specific in your questions. "What is the average sales by region?" works better than "Tell me about sales."

## Troubleshooting

### Common Error 1: "agg function failed"
**Cause**: Trying to calculate mean/sum on text columns

**Solution**: Be specific about numeric columns. Example: "What is the average of the Price column?" instead of "What is the average?"

### Common Error 2: "Could not decode CSV file"
**Cause**: CSV encoding issues

**Solution**: The agent tries multiple encodings automatically. If it still fails, save your CSV as UTF-8 in Excel/Google Sheets and try again.

### Common Error 3: "API Key not found"
**Cause**: OpenAI API key not set

**Solution**: 
* Enter key in sidebar, OR
* Create `.env` file with `OPENAI_API_KEY=your_key_here`, OR
* Set environment variable: `export OPENAI_API_KEY=your_key`

### Common Error 4: "Rate limit exceeded"
**Cause**: Too many API requests

**Solution**: Wait 60 seconds and try again. Consider upgrading your OpenAI plan for higher limits.

### Common Error 5: Charts not displaying
**Cause**: matplotlib backend issues

**Solution**: Restart the Streamlit app. Charts are generated as base64 images and should display automatically.

### Common Error 6: "No module named 'langgraph'"
**Cause**: Missing dependencies

**Solution**: 
```bash
pip install -r requirements.txt
```

## Tech Stack

* **LangGraph** - Agentic workflow orchestration with state management
* **LangChain** - LLM integration and prompt management
* **OpenAI GPT-4o** - Natural language understanding and code generation
* **Streamlit** - Interactive web interface
* **Pandas** - Data manipulation and analysis
* **Matplotlib** - Visualization generation
* **Python 3.8+** - Core language with async/await support

---

