import asyncio
import os
import streamlit as st
from typing import TypedDict, Annotated
import operator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

# Page config
st.set_page_config(
    page_title="Zerodha MCP Agent - Portfolio Analyzer", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>📊 Zerodha MCP Agent - Portfolio Analyzer</h1>", unsafe_allow_html=True)
st.markdown("AI-powered analysis of your Zerodha holdings with actionable insights")

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 Configuration Status")
    
    # Check environment variables
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_kite_key = bool(os.getenv("KITE_API_KEY"))
    has_kite_secret = bool(os.getenv("KITE_API_SECRET"))
    has_kite_token = bool(os.getenv("KITE_ACCESS_TOKEN"))
    
    st.markdown(f"{'✅' if has_openai else '❌'} OpenAI API Key")
    st.markdown(f"{'✅' if has_kite_key else '❌'} Kite API Key")
    st.markdown(f"{'✅' if has_kite_secret else '❌'} Kite API Secret")
    st.markdown(f"{'✅' if has_kite_token else '❌'} Kite Access Token")
    
    st.markdown("---")
    st.markdown("### 📈 What This Does")
    st.markdown("""
    1. **Connects** to your Zerodha account
    2. **Fetches** your current holdings
    3. **Analyzes** your portfolio using AI
    4. **Provides** actionable recommendations
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Analysis Includes")
    st.markdown("""
    - Portfolio summary & total value
    - Sector distribution
    - Diversification score
    - Investment recommendations
    - Risk assessment
    """)
    
    st.markdown("---")
    st.caption("⚠️ Access tokens expire at 3:30 PM IST daily")

# State definition
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    holdings: dict
    analysis: str

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.kite = None
    st.session_state.llm = None
    st.session_state.graph = None
    st.session_state.is_processing = False
    st.session_state.last_result = None
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

# Setup function
def setup_agent():
    """Initialize KiteConnect and LLM"""
    if not st.session_state.initialized:
        try:
            # Initialize KiteConnect
            st.session_state.kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
            st.session_state.kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))
            
            # Initialize LLM
            st.session_state.llm = ChatOpenAI(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Create graph
            st.session_state.graph = create_graph()
            
            st.session_state.initialized = True
            return None
        except Exception as e:
            return f"Initialization error: {str(e)}"
    return None

# Agent nodes
async def authenticate_node(state: AgentState) -> AgentState:
    """Verify Zerodha connection"""
    try:
        profile = st.session_state.kite.profile()
        st.success(f"✅ Connected as: {profile['user_name']} ({profile['email']})")
    except Exception as e:
        st.error(f"❌ Authentication failed: {e}")
    return state

async def fetch_holdings_node(state: AgentState) -> AgentState:
    """Fetch portfolio holdings"""
    try:
        holdings = st.session_state.kite.holdings()
        st.info(f"📦 Fetched {len(holdings)} holdings")
        return {**state, "holdings": holdings}
    except Exception as e:
        st.error(f"❌ Error fetching holdings: {e}")
        return {**state, "holdings": f"Error: {str(e)}"}

async def analyze_node(state: AgentState) -> AgentState:
    """Analyze portfolio using LLM"""
    holdings = state["holdings"]
    
    if isinstance(holdings, str) and "Error" in holdings:
        return {**state, "analysis": holdings, "messages": []}

    prompt = f"""
    You are a portfolio analyst for Indian stock markets.

    Analyze the following Zerodha holdings and provide:
    
    ## 📊 Portfolio Summary
    - Total number of stocks
    - Total investment value
    - Top 3 holdings by value
    
    ## 🏢 Sector Analysis
    - Key sectors represented
    - Sector concentration
    
    ## 📈 Diversification Score
    - Rate from 1-10 (10 = highly diversified)
    - Justification for the score
    
    ## 💡 Recommendations
    - 3 actionable recommendations
    - Risk considerations
    
    Format your response with clear sections and emojis for readability.
    
    Holdings Data:
    {holdings}
    """

    response = await st.session_state.llm.ainvoke(prompt)
    
    return {
        **state,
        "analysis": response.content,
        "messages": [response],
    }

# Create graph
def create_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("authenticate", authenticate_node)
    workflow.add_node("fetch_holdings", fetch_holdings_node)
    workflow.add_node("analyze", analyze_node)
    
    workflow.set_entry_point("authenticate")
    workflow.add_edge("authenticate", "fetch_holdings")
    workflow.add_edge("fetch_holdings", "analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()

# Main agent runner
async def run_portfolio_analyzer():
    """Execute the portfolio analysis workflow"""
    try:
        # Setup agent
        error = setup_agent()
        if error:
            return f"❌ {error}"
        
        # Run graph
        initial_state = {
            "messages": [],
            "holdings": {},
            "analysis": "",
        }
        
        result = await st.session_state.graph.ainvoke(initial_state)
        return result["analysis"]
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Button callback
def start_analysis():
    st.session_state.is_processing = True

# Main interface
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.button(
        "🚀 Analyze My Portfolio",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_processing,
        on_click=start_analysis,
    )

# Process analysis
if st.session_state.is_processing:
    with st.spinner("🔄 Analyzing your portfolio..."):
        with st.expander("📋 Processing Steps", expanded=True):
            result = st.session_state.loop.run_until_complete(run_portfolio_analyzer())
    
    st.session_state.last_result = result
    st.session_state.is_processing = False
    st.rerun()

# Display results
if st.session_state.last_result:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Display in a nice container
    with st.container():
        st.markdown(st.session_state.last_result)
    
    # Download button
    st.download_button(
        label="📥 Download Analysis",
        data=st.session_state.last_result,
        file_name="portfolio_analysis.md",
        mime="text/markdown"
    )

else:
    # Help section
    st.markdown("---")
    st.markdown("""
        <div style='padding: 25px; background-color: #f0f2f6; border-radius: 15px; margin-top: 20px;'>
        <h3>🚀 Getting Started</h3>
        <ol style='font-size: 1.1rem;'>
            <li><strong>Setup credentials</strong> in your <code>.env</code> file:
                <ul>
                    <li>OPENAI_API_KEY</li>
                    <li>KITE_API_KEY</li>
                    <li>KITE_API_SECRET</li>
                    <li>KITE_ACCESS_TOKEN</li>
                </ul>
            </li>
            <li><strong>Click</strong> "Analyze My Portfolio" button above</li>
            <li><strong>Wait</strong> for the AI to analyze your holdings</li>
            <li><strong>Review</strong> the personalized recommendations</li>
        </ol>
        
        <h4>🔐 Security Note</h4>
        <p>All credentials are stored locally in your .env file. Your data never leaves your machine except for API calls to OpenAI and Zerodha.</p>
        
        <h4>⏰ Token Expiry</h4>
        <p>Zerodha access tokens expire daily at 3:30 PM IST. Generate a new token if you see authentication errors.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
    Built with ❤️ using Streamlit, LangGraph, and Zerodha Kite API
    </div>
""", unsafe_allow_html=True)