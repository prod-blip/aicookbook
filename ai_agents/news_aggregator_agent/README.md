# 📰 News Aggregator Agent

An AI-powered news aggregator that fetches live Indian headlines and provides deep-dive analysis using autonomous agents.

✨ **Built with LangGraph for agentic workflows and GPT-4o for intelligent analysis**

## Features

* 🔄 Fetch live Indian news headlines from NewsData.io API
* 🔍 Deep dive into any article with web search for additional context
* 🤖 AI-powered analysis and summary generation using GPT-4o
* 🏷️ Filter news by categories (business, politics, sports, tech, etc.)
* ➕ Load more articles with automatic deduplication

## Setup

### Requirements

* Python 3.8+
* NewsData.io API key (free tier available)
* OpenAI API key

### Installation

1. Clone this repository:

```bash
git clone clone https://github.com/prod-blip/aicookbook.git
cd aicookbook/ai_agents/news_aggregator_agent
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Get your API credentials:
   * NewsData.io: https://newsdata.io/register (free tier: 200 requests/day)
   * OpenAI: https://platform.openai.com/api-keys

4. Setup your `.env` file:

```bash
NEWS_API_KEY=your_newsdata_io_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Running the App

1. Start the Streamlit application:

```bash
streamlit run main.py
```

2. Open your browser at `http://localhost:8501`

3. Click **Fetch** to load headlines, then click **Dive** on any article for AI analysis

## Agent Architecture

The application uses **LangGraph** with two separate graphs for different workflows:

```
┌─────────────────────────────────────────────────────────────┐
│                    NEWS AGGREGATOR AGENT                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FETCH GRAPH (Headlines)                                    │
│  ┌──────────────┐                                           │
│  │  fetch_news  │ ──────────────────────────────▶ END       │
│  └──────────────┘                                           │
│        │                                                    │
│        ▼                                                    │
│  • Calls NewsData.io API                                    │
│  • Fetches top 10 Indian headlines                          │
│  • Prepends to existing articles (latest first)             │
│  • Re-indexes all articles                                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DEEP DIVE GRAPH (Analysis)                                 │
│  ┌──────────────┐      ┌─────────────┐                      │
│  │  deep_dive   │ ────▶│   analyze   │ ────────────▶ END    │
│  └──────────────┘      └─────────────┘                      │
│        │                     │                              │
│        ▼                     ▼                              │
│  • Searches DuckDuckGo      • Builds context from           │
│  • Finds 5 related            original + related articles   │
│    articles                 • Sends to GPT-4o               │
│  • Extracts snippets        • Generates 150-200 word        │
│                               summary with key facts        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### State Definition

```python
class AgentState(TypedDict):
    all_news: List[Dict]        # All fetched articles
    selected_news: Dict         # User's selected article
    deep_dive_data: List[Dict]  # Related articles from web search
    analysis: str               # Final AI summary
    error: str                  # Error message (if any)
```

### Node Responsibilities

1. **fetch_news_node**: Calls NewsData.io API, fetches top 10 Indian headlines in English, prepends new articles to existing list, re-indexes all articles

2. **deep_dive_node**: Takes selected article title, searches DuckDuckGo for related context (5 results), extracts titles and snippets

3. **analyze_node**: Combines original article + deep dive results, sends to GPT-4o with structured prompt, returns concise summary with "What Happened", "Key Facts", and "Why It Matters"

## Important Notes

⚠️ **API Limits**: NewsData.io free tier allows 200 requests/day. Each "Fetch" or "More" click uses 1 request.

🔐 **API Keys**: Never commit your `.env` file. Use `.env.example` as a template.

💰 **OpenAI Costs**: Each deep dive analysis uses ~500-800 tokens. Monitor your usage at https://platform.openai.com/usage

📊 **Categories**: Category filtering happens client-side after fetching. All categories are fetched, then filtered in UI.

🌐 **Search Region**: DuckDuckGo searches are configured for India (region: `in-en`) for relevant results.

## Troubleshooting

### "News API Missing" in sidebar
* Ensure `NEWS_API_KEY` is set in your `.env` file
* Verify the key is valid at https://newsdata.io/account

### "OpenAI Missing" in sidebar
* Ensure `OPENAI_API_KEY` is set in your `.env` file
* Check you have credits/quota available

### No news appearing after Fetch
* Check your internet connection
* Verify NewsData.io API key is valid
* Check terminal for error messages

### Deep dive returns irrelevant results
* DuckDuckGo search depends on article title quality
* Some niche topics may have limited web coverage

### Rate limit errors
* NewsData.io free tier: 200 requests/day
* Wait for daily reset or upgrade plan

## Tech Stack

* **Agent Framework**: LangGraph
* **LLM**: OpenAI GPT-4o
* **News API**: NewsData.io
* **Web Search**: DuckDuckGo (via duckduckgo-search)
* **Frontend**: Streamlit
* **Async**: Python asyncio

---
