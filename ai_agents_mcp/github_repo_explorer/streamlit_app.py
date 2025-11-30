"""
GitHub Repository Explorer - Streamlit Application

AI-powered GitHub repository exploration using natural language queries
via GitHub's Official Model Context Protocol (MCP) server and LangGraph.
"""

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from agent import run_agent

load_dotenv()


# ============= SESSION STATE INITIALIZATION =============

def init_session_state():
    """Initialize all session state variables"""

    # Event loop - ONLY ONCE
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)

    # Processing flags
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    # Results
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    # Input values
    if 'repo_url' not in st.session_state:
        st.session_state.repo_url = ""

    if 'github_token' not in st.session_state:
        st.session_state.github_token = ""

    if 'query' not in st.session_state:
        st.session_state.query = ""


# ============= PAGE CONFIGURATION =============

st.set_page_config(
    page_title="GitHub Repository Explorer",
    page_icon="🔍",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>🔍 GitHub Repository Explorer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Explore any GitHub repository with AI-powered analysis using GitHub's Official MCP Server</p>",
    unsafe_allow_html=True
)


# ============= SIDEBAR =============

with st.sidebar:
    st.markdown("### 🔑 Configuration Status")

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_github = bool(os.getenv("GITHUB_TOKEN"))

    st.markdown(f"{'✅' if has_openai else '❌'} OpenAI API Key")
    st.markdown(f"{'✅' if has_github else '⚠️'} GitHub Token (optional)")

    if not has_openai:
        st.warning("⚠️ Please set OPENAI_API_KEY in .env file")

    st.markdown("---")
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. **Enter repository URL** (any public GitHub repo)
    2. **Ask your question** in natural language
    3. **MCP server explores** the repository
    4. **AI analyzes** and explains findings
    5. **View results** with code snippets
    """)

    st.markdown("---")
    st.markdown("### 🎯 Example Queries")
    st.markdown("""
    - "How is authentication implemented?"
    - "What's the overall project structure?"
    - "Show me the database models"
    - "How does error handling work?"
    - "Find all API endpoints"
    - "Explain the state management"
    """)

    st.markdown("---")
    st.markdown("### 💡 Features")
    st.markdown("""
    🤖 GitHub's Official MCP Server
    🔍 Natural language queries
    📊 Repository insights & analytics
    🔄 Issues & Pull Requests
    ⚡ Real-time GitHub API access
    🎨 AI-powered explanations
    🐳 Runs via Docker
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Requirements")
    st.markdown("""
    - Docker installed and running
    - GitHub Personal Access Token
    - OpenAI API Key
    """)


# ============= INITIALIZE SESSION STATE =============

init_session_state()


# ============= INPUT FORM =============

st.markdown("### 🔍 Explore Repository")

col1, col2 = st.columns([2, 1])

with col1:
    repo_url = st.text_input(
        "📦 GitHub Repository URL",
        placeholder="https://github.com/fastapi/fastapi",
        help="Enter the full URL of the GitHub repository",
        key="repo_url_input",
        value=st.session_state.repo_url
    )

with col2:
    github_token = st.text_input(
        "🔑 GitHub Token (optional)",
        type="password",
        help="Personal access token for higher rate limits",
        key="github_token_input",
        value=st.session_state.github_token
    )

query = st.text_area(
    "❓ What would you like to know about this repository?",
    placeholder="e.g., How does dependency injection work in this project?",
    height=100,
    key="query_input",
    value=st.session_state.query
)


# ============= BUTTON CALLBACK =============

def start_exploration():
    """Start repository exploration"""
    if not repo_url or not query:
        st.error("❌ Please enter both repository URL and query")
        return
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OpenAI API key not found in .env file")
        return

    # Store values
    st.session_state.repo_url = repo_url
    st.session_state.github_token = github_token
    st.session_state.query = query
    st.session_state.is_processing = True


# ============= EXPLORE BUTTON =============

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.button(
        "🚀 Explore Repository",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_processing,
        on_click=start_exploration
    )


# ============= PROCESS EXPLORATION =============

if st.session_state.is_processing:
    with st.spinner("🔄 Exploring repository..."):
        with st.expander("📋 Agent Progress", expanded=True):
            st.write("⏳ Starting MCP server...")
            st.write("🤖 Planning tool calls...")
            st.write("🔍 Exploring repository...")
            st.write("📊 Analyzing code...")
            st.write("💡 Generating explanation...")

            # Run agent
            result = st.session_state.loop.run_until_complete(
                run_agent(
                    repo_url=st.session_state.repo_url,
                    github_token=st.session_state.github_token or os.getenv("GITHUB_TOKEN", ""),
                    user_query=st.session_state.query
                )
            )

    st.session_state.last_result = result
    st.session_state.is_processing = False
    st.rerun()


# ============= DISPLAY RESULTS =============

if st.session_state.last_result:
    st.markdown("---")
    st.markdown("## 🎯 Analysis Results")

    result = st.session_state.last_result

    # Two column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💡 Explanation")
        with st.container():
            st.markdown(result["explanation"])

    with col2:
        st.markdown("### 📊 Summary")

        # Extract repo name from URL
        repo_name = result['repo_url'].split('github.com/')[-1] if 'github.com/' in result['repo_url'] else result['repo_url']

        st.info(f"""
**Repository**: `{repo_name}`

**Query**: {result['user_query'][:100]}{'...' if len(result['user_query']) > 100 else ''}

**MCP Tools Used**: {len(result['tool_results'])}

**Available Tools**: {len(result.get('available_tools', []))}

**Status**: {'✅ Success' if result.get('success') else '⚠️ Partial'}
        """)

        # Show tool results summary
        if result['tool_results']:
            st.markdown("**🔧 MCP Tools Used:**")
            for tool_result in result['tool_results']:
                tool_name = tool_result.get('tool', 'unknown')
                success = tool_result.get('success', False)
                status_emoji = "✅" if success else "❌"
                st.markdown(f"{status_emoji} `{tool_name}`")

    # MCP Tool Results Details (optional expander)
    if result['tool_results']:
        st.markdown("---")
        with st.expander("🔍 View Raw MCP Tool Results", expanded=False):
            for idx, tool_result in enumerate(result['tool_results'], 1):
                st.markdown(f"### Tool {idx}: {tool_result.get('tool', 'unknown')}")
                if tool_result.get('success'):
                    st.json(tool_result.get('result', {}))
                else:
                    st.error(f"Error: {tool_result.get('error', 'Unknown error')}")
                st.markdown("---")

    # Errors section
    if result.get('errors'):
        st.markdown("---")
        st.warning("⚠️ Some operations encountered issues:")
        for error in result['errors']:
            st.error(error)

    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Download button
        st.download_button(
            label="📥 Download Analysis",
            data=result["explanation"],
            file_name=f"github_analysis_{repo_name.replace('/', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col3:
        # New query button
        def reset_query():
            st.session_state.last_result = None
            st.session_state.query = ""

        st.button(
            "🔄 New Query",
            use_container_width=True,
            on_click=reset_query
        )

else:
    # Help section - show when no results
    st.markdown("---")
    st.markdown("""
        <div style='padding: 25px; background-color: #f0f2f6; border-radius: 15px;'>
        <h3>🚀 Getting Started</h3>
        <ol>
            <li><strong>Enter repository URL:</strong> Any public GitHub repository</li>
            <li><strong>Optional token:</strong> Add GitHub token for higher rate limits (5000 vs 60 requests/hour)</li>
            <li><strong>Ask your question:</strong> Use natural language to explore the codebase</li>
            <li><strong>Get insights:</strong> AI analyzes code and provides explanations with snippets</li>
        </ol>

        <h4>📝 Example</h4>
        <ul>
            <li><strong>URL:</strong> https://github.com/fastapi/fastapi</li>
            <li><strong>Query:</strong> "How does FastAPI handle dependency injection?"</li>
        </ul>

        <h4>🔐 GitHub Token (Optional)</h4>
        <p>Create a Personal Access Token at <a href="https://github.com/settings/tokens" target="_blank">github.com/settings/tokens</a></p>
        <p><strong>Permissions needed:</strong> public_repo (or repo for private repositories)</p>
        <p><strong>Benefits:</strong></p>
        <ul>
            <li>✅ 5000 requests/hour (vs 60 without token)</li>
            <li>✅ Access to private repositories (if permission granted)</li>
            <li>✅ Fewer rate limit issues</li>
        </ul>

        <h4>🎯 What You Can Ask</h4>
        <ul>
            <li><strong>Architecture:</strong> "What's the overall project structure?"</li>
            <li><strong>Implementation:</strong> "How is authentication implemented?"</li>
            <li><strong>Features:</strong> "Find all API endpoints in this project"</li>
            <li><strong>Code Search:</strong> "Show me files that handle database queries"</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)


# ============= FOOTER =============

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Built with ❤️ using LangGraph, GPT-4o & GitHub's Official MCP Server</p>",
    unsafe_allow_html=True
)
