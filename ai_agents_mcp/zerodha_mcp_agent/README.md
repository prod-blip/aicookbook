# 📊 Zerodha Portfolio Analyzer

A Streamlit application that analyzes your Zerodha holdings using Kite Model Context Protocol (MCP) & AI to provide personalized portfolio insights, diversification scores, and actionable investment recommendations.

✨ **Powered by LangGraph and OpenAI GPT-4o for intelligent portfolio analysis!**

https://github.com/user-attachments/assets/c72b14c5-ae48-4965-970d-18adc4426535

## Features

* **AI-Powered Analysis**: Get comprehensive portfolio insights using advanced language models
* **Real-time Holdings**: Fetch your current Zerodha holdings directly via Kite API
* **Sector Analysis**: Understand your portfolio's sector distribution and concentration
* **Diversification Score**: Receive a quantitative assessment of portfolio diversification (1-10 scale)
* **Actionable Recommendations**: Get 3 personalized recommendations to improve your portfolio
* **Interactive UI**: User-friendly Streamlit interface with status indicators and progress tracking
* **Export Results**: Download your analysis as a markdown file for future reference

## Setup

### Requirements

* Python 3.8+
* OpenAI API Key
* Zerodha Kite Connect API credentials (API Key, API Secret)
* Active Zerodha trading account

### Installation

1. Clone this repository:

```bash
git clone https://github.com/prod-blip/aicookbook.git
cd aicookbook/ai_agents_mcp/zerodha_mcp_agent
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit
langchain-openai
langchain-core
langgraph
kiteconnect
python-dotenv
```

3. Get your API credentials:
   * **OpenAI API Key**: Get from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   * **Kite API Key & Secret**: Create an app at [developers.kite.trade](https://developers.kite.trade/)
     * Login to Kite Developer Console
     * Create a new app (set redirect URL as `http://127.0.0.1`)
     * Note down your API Key and API Secret

4. Setup your `.env` file:

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
KITE_API_KEY=your_kite_api_key_here
KITE_API_SECRET=your_kite_api_secret_here
KITE_ACCESS_TOKEN=your_access_token_here
```

### Getting Zerodha Access Token

Access tokens need to be generated daily (valid until 3:30 PM IST). Run this one-time setup:

1. Run the access token script:

```bash
python accesstoken.py
```

2. Follow the prompts:
   * Click the generated URL
   * Login to Zerodha
   * Copy the `request_token` from the redirect URL
   * Paste it in the terminal
   * Copy the generated access token to your `.env` file

## Running the App

1. Start the Streamlit app:

```bash
streamlit run zerodha_mcp_agent.py
```

2. In the app interface:
   * Verify all credentials are configured (check sidebar status)
   * Click "🚀 Analyze My Portfolio"
   * Wait for the analysis to complete
   * Review your personalized portfolio insights
   * Optionally download the analysis as a markdown file

## Analysis Output

The AI provides comprehensive analysis including:

### 📊 Portfolio Summary
* Total number of stocks
* Total investment value
* Top 3 holdings by value

### 🏢 Sector Analysis
* Key sectors represented in your portfolio
* Sector concentration levels
* Industry distribution

### 📈 Diversification Score
* Numerical score from 1-10 (10 = highly diversified)
* Detailed justification for the score
* Risk concentration insights

### 💡 Recommendations
* 3 specific, actionable recommendations
* Risk considerations
* Suggestions for portfolio improvement

## Agent Architecture

The application uses a **LangGraph-based agentic workflow** with three sequential nodes:

```
authenticate → fetch_holdings → analyze
```

1. **Authenticate Node**: Verifies connection to Zerodha Kite API
2. **Fetch Holdings Node**: Retrieves current portfolio holdings
3. **Analyze Node**: Sends holdings to GPT-4o for AI-powered analysis

## Important Notes

⚠️ **Token Expiry**: Zerodha access tokens expire daily at 3:30 PM IST. You'll need to regenerate the token using `accesstoken.py` each day.

🔐 **Security**: All credentials are stored locally in your `.env` file. API calls are made only to OpenAI and Zerodha's official servers.


📊 **Data Freshness**: Holdings data is fetched in real-time from your Zerodha account.

## Troubleshooting

### "Incorrect api_key or access_token" Error
* Verify your access token is current (regenerate if expired)
* Ensure no extra quotes or spaces in `.env` file
* Check that you're using the access token, not the request token

### "Authentication Failed"
* Verify Kite API credentials are correct
* Ensure your Kite Connect app is approved
* Check if subscription is active

### Docker Not Required
Unlike some MCP agents, this application directly uses the KiteConnect Python library and doesn't require Docker.

## Tech Stack

* **Frontend**: Streamlit
* **Agent Framework**: LangGraph
* **LLM**: OpenAI GPT-4o
* **Trading API**: Zerodha Kite Connect
* **Async Processing**: Python asyncio


