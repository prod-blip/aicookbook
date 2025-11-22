import os
import asyncio
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

import streamlit as st
import requests
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# ----------------------------
# SIMPLE ASYNC RUN HELPER
# ----------------------------
def run_async(coro):
    """
    Run an async coroutine safely from Streamlit.
    Tries to reuse existing loop; falls back to asyncio.run.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If the loop is already running (rare in Streamlit),
            # create a new temporary loop for the call.
            return asyncio.run(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ----------------------------
# STATE DEFINITION (trimmed)
# ----------------------------
class AgentState(TypedDict):
    """Minimal state for the news agent"""
    all_news: List[Dict]          # All fetched articles
    selected_news: Dict           # Selected article
    deep_dive_data: List[Dict]    # Related articles from web search
    analysis: str                 # Final summary / report
    error: str                    # Error message (if any)


# ----------------------------
# CONFIG
# ----------------------------
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0.3
)


# ----------------------------
# NODE: FETCH NEWS
# ----------------------------
async def fetch_news_node(state: AgentState) -> AgentState:
    """
    Fetch top headlines from newsdata.io and prepend newest articles.
    Returns updated state with combined 'all_news' list or error.
    """
    try:
        url = "https://newsdata.io/api/1/latest"
        params = {
            "apikey": NEWS_API_KEY,
            "country": "in",
            "language": "en",
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("results", [])[:10]

        new_articles = []
        for art in articles:
            new_articles.append({
                "title": art.get("title", "No title"),
                "description": art.get("description", "No description"),
                "source": art.get("source_id", "Unknown"),
                "url": art.get("link", ""),
                "publishedAt": art.get("pubDate", ""),
                "category": art.get("category", []),
            })

        existing = state.get("all_news", []) or []
        combined = new_articles + existing

        # re-index
        for idx, a in enumerate(combined):
            a["index"] = idx + 1

        return {
            **state,
            "all_news": combined,
            "error": "",
        }

    except Exception as e:
        return {
            **state,
            "error": str(e),
        }


# ----------------------------
# NODE: DEEP DIVE (web search)
# ----------------------------
async def deep_dive_node(state: AgentState) -> AgentState:
    """
    Use DuckDuckGo (via DDGS) to find related links/snippets for selected article.
    """
    selected = state.get("selected_news", {})
    if not selected:
        return {**state, "deep_dive_data": [], "error": "No article selected"}

    try:
        query = f"{selected.get('title', '')} India news"
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=query, region="in-en", max_results=5))

        formatted = []
        for i, r in enumerate(results):
            formatted.append({
                "index": i + 1,
                "title": r.get("title", "No title"),
                "snippet": r.get("body", "No snippet"),
                "url": r.get("href", ""),
            })

        return {**state, "deep_dive_data": formatted, "error": ""}

    except Exception as e:
        return {**state, "deep_dive_data": [], "error": str(e)}


# ----------------------------
# NODE: ANALYZE WITH LLM
# ----------------------------
async def analyze_node(state: AgentState) -> AgentState:
    """
    Use the LLM to synthesize a concise report using the selected article
    and deep-dive results.
    """
    if state.get("error"):
        return {**state, "analysis": f"❌ Error: {state['error']}"}

    selected = state.get("selected_news", {})
    deep = state.get("deep_dive_data", [])

    if not selected:
        return {**state, "analysis": "❌ No article selected"}

    try:
        ctx_lines = [
            f"ORIGINAL NEWS:",
            f"Title: {selected.get('title','')}",
            f"Source: {selected.get('source','')}",
            f"Description: {selected.get('description','')}",
            "",
            "RELATED ARTICLES:"
        ]
        for a in deep:
            ctx_lines.append(f"{a['index']}. {a['title']}\n   {a['snippet']}")

        context = "\n".join(ctx_lines)

        prompt = f"""Analyze this news and provide a brief summary:

{context}

Include:
1. What Happened: 2-3 sentences
2. Key Facts: Bullet or short list
3. Why It Matters: Context & significance

Keep it concise (150-200 words).
"""

        response = await llm.ainvoke(prompt)
        return {**state, "analysis": response.content, "error": ""}

    except Exception as e:
        return {**state, "analysis": f"❌ Error: {str(e)}", "error": str(e)}


# ----------------------------
# GRAPH CREATION (single, top-level)
# ----------------------------
def create_fetch_graph():
    g = StateGraph(AgentState)
    g.add_node("fetch_news", fetch_news_node)
    g.set_entry_point("fetch_news")
    g.add_edge("fetch_news", END)
    return g.compile()


def create_deep_dive_graph():
    g = StateGraph(AgentState)
    g.add_node("deep_dive", deep_dive_node)
    g.add_node("analyze", analyze_node)
    g.set_entry_point("deep_dive")
    g.add_edge("deep_dive", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


FETCH_GRAPH = create_fetch_graph()
DEEP_DIVE_GRAPH = create_deep_dive_graph()


# ----------------------------
# STREAMLIT UI + SESSION STATE
# ----------------------------
st.set_page_config(page_title="News Aggregator AI", page_icon="📰", layout="wide")

# minimal session state
if "all_news" not in st.session_state:
    st.session_state.all_news = []
if "selected_news" not in st.session_state:
    st.session_state.selected_news = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []

# UI: sidebar
with st.sidebar:
    st.title("📰 News AI")
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("News API\n\n**:green[Active]**" if NEWS_API_KEY else "News API\n\n**:red[Missing]**")
    with col2:
        st.markdown("OpenAI\n\n**:green[Active]**" if OPENAI_API_KEY else "OpenAI\n\n**:red[Missing]**")

    st.divider()
    st.subheader("Categories")
    available_categories = [
        "business", "crime", "domestic", "education", "entertainment",
        "environment", "food", "health", "lifestyle", "other",
        "politics", "science", "sports", "technology", "top",
        "tourism", "world"
    ]
    st.session_state.selected_categories = st.multiselect(
        "Select categories (max 5)",
        options=available_categories,
        default=st.session_state.selected_categories,
        max_selections=5
    )
    st.divider()
    st.info("Built with LangGraph & Streamlit")


# HEADER + ACTIONS
col_title, col_actions = st.columns([3, 1])
with col_title:
    st.title("News Aggregator")
    st.markdown("Live Indian headlines powered by Autonomous AI agents.")
with col_actions:
    fetch_btn = st.button("🔄 Fetch", type="primary", use_container_width=True)
    more_btn = st.button("➕ More", use_container_width=True, disabled=not st.session_state.all_news)


# SHARED fetch function (used by Fetch and More)
def fetch_latest():
    initial_state: AgentState = {
        "all_news": st.session_state.all_news,
        "selected_news": {},
        "deep_dive_data": [],
        "analysis": "",
        "error": "",
    }
    result = run_async(FETCH_GRAPH.ainvoke(initial_state))
    return result


# HANDLE fetch / more
if fetch_btn:
    with st.status("🔄 Fetching latest headlines...", expanded=True) as status:
        res = fetch_latest()
        if res.get("error"):
            status.update(label="❌ Fetch failed", state="error")
            st.error(f"Error: {res['error']}")
        else:
            st.session_state.all_news = res["all_news"]
            st.session_state.analysis = None
            st.session_state.selected_news = None
            status.update(label="✅ News fetched", state="complete")
            st.rerun()

if more_btn:
    with st.status("➕ Loading more articles...", expanded=True) as status:
        res = fetch_latest()
        if res.get("error"):
            st.error(f"Error: {res['error']}")
        else:
            st.session_state.all_news = res["all_news"]
            status.update(label="✅ Loaded more articles", state="complete")
            st.rerun()


# DEEP DIVE PROCESS (triggered when user clicks Dive on a card)
if st.session_state.is_processing and st.session_state.selected_news:
    with st.status(f"🕵️‍♂️ Deep diving: {st.session_state.selected_news.get('title','')[:40]}...", expanded=True) as status:
        state: AgentState = {
            "all_news": st.session_state.all_news,
            "selected_news": st.session_state.selected_news,
            "deep_dive_data": [],
            "analysis": "",
            "error": "",
        }
        res = run_async(DEEP_DIVE_GRAPH.ainvoke(state))
        if res.get("error"):
            status.update(label="❌ Analysis failed", state="error")
            st.error(f"Error: {res['error']}")
        else:
            st.session_state.analysis = res.get("analysis", "")
            status.update(label="✅ Report Ready!", state="complete")
        st.session_state.is_processing = False
        st.rerun()


# DISPLAY analysis (if present)
if st.session_state.analysis and st.session_state.selected_news:
    st.divider()
    col_a1, col_a2 = st.columns([3, 1])
    with col_a1:
        st.subheader("📊 AI Intelligence Report")
        st.caption(f"Subject: {st.session_state.selected_news['title']}")
    with col_a2:
        st.download_button(
            label="📥 Download Report",
            data=f"# {st.session_state.selected_news['title']}\n\n{st.session_state.analysis}",
            file_name="news_analysis.md",
            mime="text/markdown",
            use_container_width=True
        )
    with st.container():
        st.markdown(st.session_state.analysis)
        st.markdown(f"**Original Source:** [{st.session_state.selected_news.get('source','')}]({st.session_state.selected_news.get('url','')})")


# NEWS FEED
st.divider()
if not st.session_state.all_news:
    st.info("👆 Click **Fetch** to start reading the news.")
else:
    # filter by selected categories (UI-only)
    filtered = st.session_state.all_news
    if st.session_state.selected_categories:
        filtered = [
            n for n in st.session_state.all_news
            if any(cat in (n.get("category") or []) for cat in st.session_state.selected_categories)
        ]

    st.subheader(f"📋 Latest Feed ({len(filtered)} articles)")

    for news in filtered:
        with st.container():
            col_content, col_action = st.columns([5, 1])
            with col_content:
                st.markdown(f"##### {news['index']}. {news['title']}")
                cats = ", ".join(news.get("category", [])) if news.get("category") else "N/A"
                st.caption(f"📅 {news.get('publishedAt','N/A')} | 🔗 {news.get('source','N/A')} | 🏷️ {cats}")
                if news.get("description"):
                    with st.expander("Read snippet"):
                        st.write(news["description"])
            with col_action:
                if st.button("🔍 Dive", key=f"dive_{news['index']}", use_container_width=True):
                    st.session_state.selected_news = news
                    st.session_state.is_processing = True
                    st.rerun()

# FOOTER
st.markdown("---")
st.markdown("<center><small>Powered by NewsData.io & DuckDuckGo</small></center>", unsafe_allow_html=True)
