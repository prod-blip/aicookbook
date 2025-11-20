import os
import pandas as pd
from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv
import streamlit as st
import asyncio

load_dotenv()

# ============================================
# AGENT CODE
# ============================================

# ============================================
# State Definition
# ============================================

class AnalysisState(TypedDict):
    messages: Annotated[list, operator.add]
    file_path: str                    # Path to uploaded file
    df: Optional[pd.DataFrame]        # Loaded dataframe
    user_query: str                   # User's question
    analysis_plan: str                # What analysis to do
    analysis_code: str                # Python code to execute
    analysis_result: dict             # Results from analysis
    chart_needed: bool                # Whether chart is needed
    chart_base64: Optional[str]       # Chart as base64
    final_response: str               # Final answer to user


# ============================================
# Node 1: Load Data
# ============================================

async def load_data_node(state: AnalysisState) -> AnalysisState:
    """
    Loads CSV/Excel file into DataFrame.
    
    Reads: state["file_path"]
    Updates: state["df"]
    """
    print("\n=== Loading Data ===")
    
    try:
        file_path = state["file_path"]

        # Detect file type and load
        if file_path.endswith('.csv'):
            # Try multiple encodings for CSV files
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ Successfully loaded with {encoding} encoding")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if df is None:
                raise ValueError("Could not decode CSV file with any standard encoding")

        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return {
            **state,
            "df": df,
        }
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return {
            **state,
            "df": None,
            "final_response": f"Error loading file: {str(e)}",
        }


# ============================================
# Node 2: Understand Query
# ============================================

async def understand_query_node(state: AnalysisState) -> AnalysisState:
    """
    LLM understands user's query and creates analysis plan.
    
    Reads: state["df"], state["user_query"]
    Updates: state["analysis_plan"], state["chart_needed"]
    """
    print("\n=== Understanding Query ===")
    
    # Check if data loaded successfully
    if state["df"] is None:
        return {**state}
    
    df = state["df"]
    
    # Get LLM from environment
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create data summary
    data_summary = f"""
    Dataset Info:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Data types: {df.dtypes.to_dict()}
    - First few rows:
    {df.head(3).to_string()}
    """
    
    prompt = f"""
    You are a data analysis expert. A user has uploaded a dataset and asked a question.
    
    {data_summary}
    
    User Question: {state["user_query"]}
    
    Your task:
    1. Understand what analysis is needed
    2. Determine if a chart/visualization would help
    
    Respond in this format:
    ANALYSIS_PLAN: [Describe what analysis to perform]
    CHART_NEEDED: [YES or NO]
    """
    
    response = await llm.ainvoke(prompt)
    content = response.content
    
    # Parse response
    analysis_plan = ""
    chart_needed = False
    
    for line in content.split('\n'):
        if line.startswith('ANALYSIS_PLAN:'):
            analysis_plan = line.replace('ANALYSIS_PLAN:', '').strip()
        elif line.startswith('CHART_NEEDED:'):
            chart_needed = 'YES' in line.upper()
    
    print(f"📋 Plan: {analysis_plan}")
    print(f"📊 Chart needed: {chart_needed}")
    
    return {
        **state,
        "analysis_plan": analysis_plan,
        "chart_needed": chart_needed,
    }


# ============================================
# Node 3: Generate Analysis Code
# ============================================

async def generate_code_node(state: AnalysisState) -> AnalysisState:
    """
    LLM generates Python code to perform the analysis.
    
    Reads: state["df"], state["analysis_plan"]
    Updates: state["analysis_code"]
    """
    print("\n=== Generating Analysis Code ===")
    
    if state["df"] is None:
        return {**state}
    
    df = state["df"]
    
    # Get LLM from environment
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    prompt = f"""
    You are a Python data analysis expert. Generate code to perform this analysis:
    
    Analysis Plan: {state["analysis_plan"]}
    
    Dataset Info:
    - Shape: {df.shape}
    - Numeric columns: {numeric_cols}
    - Categorical columns: {categorical_cols}
    
    IMPORTANT RULES:
    1. Use 'df' variable (already loaded pandas DataFrame)
    2. Only calculate mean/sum/std on NUMERIC columns: {numeric_cols}
    3. For categorical columns, use value_counts() or groupby()
    4. Store results in a dictionary called 'result'
    5. Handle missing values appropriately
    6. Convert results to standard Python types (use .item() for numpy types)
    
    Example format:
```python
    result = {{}}
    # For numeric analysis
    result['average'] = df['{numeric_cols[0] if numeric_cols else 'column'}'].mean()
    # For categorical analysis  
    result['counts'] = df['{categorical_cols[0] if categorical_cols else 'column'}'].value_counts().to_dict()
```
    
    Return ONLY executable code, no explanations.
    """
    
    response = await llm.ainvoke(prompt)
    code = response.content
    
    # Extract code from markdown if present
    if '```python' in code:
        code = code.split('```python')[1].split('```')[0].strip()
    elif '```' in code:
        code = code.split('```')[1].split('```')[0].strip()
    
    print(f"💻 Generated code")
    
    return {
        **state,
        "analysis_code": code,
    }


# ============================================
# Node 4: Execute Analysis
# ============================================

async def execute_analysis_node(state: AnalysisState) -> AnalysisState:
    """
    Executes the generated analysis code.
    
    Reads: state["df"], state["analysis_code"]
    Updates: state["analysis_result"]
    """
    print("\n=== Executing Analysis ===")
    
    if state["df"] is None:
        return {**state}
    
    try:
        df = state["df"]
        code = state["analysis_code"]
        
        # Create execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'result': {}
        }
        
        # Execute code
        exec(code, exec_globals)
        result = exec_globals['result']
        
        print("✅ Analysis executed successfully")
        
        return {
            **state,
            "analysis_result": result,
        }
        
    except Exception as e:
        print(f"❌ Error executing analysis: {e}")
        return {
            **state,
            "analysis_result": {"error": str(e)},
        }


# ============================================
# Node 5: Generate Chart
# ============================================

async def generate_chart_node(state: AnalysisState) -> AnalysisState:
    """
    Generates visualization if needed.
    
    Reads: state["df"], state["chart_needed"], state["analysis_plan"]
    Updates: state["chart_base64"]
    """
    print("\n=== Generating Chart ===")
    
    if not state["chart_needed"] or state["df"] is None:
        print("⏭️  Chart not needed")
        return {**state}
    
    try:
        df = state["df"]
        
        # Get LLM from environment
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Ask LLM to generate chart code
        prompt = f"""
        Generate Python code using matplotlib to create a chart for this analysis:
        
        Analysis Plan: {state["analysis_plan"]}
        Columns: {list(df.columns)}
        
        Generate ONLY executable code that:
        1. Uses 'df' variable
        2. Creates a matplotlib figure
        3. Saves to variable 'fig'
        
        Return ONLY code, no explanations.
        """
        
        response = await llm.ainvoke(prompt)
        code = response.content
        
        # Extract code
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        # Execute chart code
        exec_globals = {
            'df': df,
            'pd': pd,
            'plt': plt,
            'fig': None
        }
        
        exec(code, exec_globals)
        fig = exec_globals.get('fig') or plt.gcf()
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        print("✅ Chart generated")
        
        return {
            **state,
            "chart_base64": chart_base64,
        }
        
    except Exception as e:
        print(f"❌ Error generating chart: {e}")
        return {
            **state,
            "chart_base64": None,
        }


# ============================================
# Node 6: Create Final Response
# ============================================

async def create_response_node(state: AnalysisState) -> AnalysisState:
    """
    Creates natural language response for user.
    
    Reads: state["user_query"], state["analysis_result"], state["chart_base64"]
    Updates: state["final_response"]
    """
    print("\n=== Creating Response ===")
    
    if state["df"] is None:
        return {**state}
    
    # Check for errors
    if "error" in state.get("analysis_result", {}):
        return {
            **state,
            "final_response": f"Error during analysis: {state['analysis_result']['error']}",
        }
    
    # Get LLM from environment
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = f"""
    Create a clear, conversational response to the user's question.
    
    User Question: {state["user_query"]}
    Analysis Results: {state["analysis_result"]}
    Chart Available: {'Yes' if state.get("chart_base64") else 'No'}
    
    Provide:
    1. Direct answer to their question
    2. Key insights from the analysis
    3. Any relevant numbers/statistics
    
    Be concise and clear. Use emojis for readability.
    """
    
    response = await llm.ainvoke(prompt)
    
    print("✅ Response created")
    
    return {
        **state,
        "final_response": response.content,
    }


# ============================================
# Build Graph
# ============================================

def create_analysis_graph():
    """Creates the LangGraph workflow"""
    
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("understand_query", understand_query_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_analysis", execute_analysis_node)
    workflow.add_node("generate_chart", generate_chart_node)
    workflow.add_node("create_response", create_response_node)
    
    # Define edges
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "understand_query")
    workflow.add_edge("understand_query", "generate_code")
    workflow.add_edge("generate_code", "execute_analysis")
    workflow.add_edge("execute_analysis", "generate_chart")
    workflow.add_edge("generate_chart", "create_response")
    workflow.add_edge("create_response", END)
    
    return workflow.compile()


# ============================================
# Test Runner
# ============================================

async def run_analysis(file_path: str, user_query: str):
    """
    Main function to run analysis
    
    Args:
        file_path: Path to CSV/Excel file
        user_query: User's question in natural language
    """
    
    graph = create_analysis_graph()
    
    initial_state = {
        "messages": [],
        "file_path": file_path,
        "df": None,
        "user_query": user_query,
        "analysis_plan": "",
        "analysis_code": "",
        "analysis_result": {},
        "chart_needed": False,
        "chart_base64": None,
        "final_response": "",
    }
    
    result = await graph.ainvoke(initial_state)
    
    return {
        "response": result["final_response"],
        "chart": result.get("chart_base64"),
        "analysis": result["analysis_result"],
    }


# ============================================
# STREAMLIT UI CODE
# ============================================

# Page config - ALWAYS FIRST
st.set_page_config(
    page_title="Data Analysis Agent",
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
    .result-box {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>📊 Data Analysis Agent</h1>", unsafe_allow_html=True)
st.markdown("Upload your data and ask questions in natural language - AI will analyze and visualize!")

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    
    # API Key input
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key or set it in .env file"
    )
    
    # Update API key
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    
    # Check API key status
    has_api_key = bool(api_key_input)
    st.markdown(f"{'✅' if has_api_key else '❌'} API Key Status")
    
    if not has_api_key:
        st.warning("⚠️ Please enter your OpenAI API key to continue")
    
    st.markdown("---")
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. **Upload** your CSV/Excel file
    2. **Ask** questions in natural language
    3. **Get** instant analysis & charts
    4. **Download** results
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - What is the average sales by region?
    - Show me the trend over time
    - Which category has highest revenue?
    - Compare performance across groups
    - Find outliers in the data
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Supported Files")
    st.markdown("• CSV files (.csv)")
    st.markdown("• Excel files (.xlsx, .xls)")
    
    st.markdown("---")
    st.caption("Built with LangGraph + GPT-4o")

# ============================================
# Initialize Session State
# ============================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.is_processing = False
    st.session_state.uploaded_file = None
    st.session_state.file_path = None
    st.session_state.last_result = None
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)
    st.session_state.analysis_history = []

# ============================================
# File Upload Section
# ============================================

st.markdown("## 📁 Upload Your Data")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your dataset to begin analysis"
)

# Handle file upload
if uploaded_file is not None:
    # Save to temp file
    if st.session_state.uploaded_file != uploaded_file.name:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.file_path = temp_path
        st.session_state.uploaded_file = uploaded_file.name
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show file info
        file_size = uploaded_file.size / 1024  # KB
        st.info(f"📊 File size: {file_size:.2f} KB")

# ============================================
# Query Input Section
# ============================================

if st.session_state.file_path:
    st.markdown("---")
    st.markdown("## 💬 Ask Your Question")
    
    # Text input for query
    user_query = st.text_input(
        "What would you like to know about your data?",
        placeholder="e.g., What is the average sales by region?",
        disabled=not has_api_key or st.session_state.is_processing,
    )
    
    # Button callback
    def start_analysis():
        if user_query:
            st.session_state.is_processing = True
        else:
            st.warning("⚠️ Please enter a question first!")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button(
            "🚀 Analyze Data",
            type="primary",
            use_container_width=True,
            disabled=not has_api_key or st.session_state.is_processing or not user_query,
            on_click=start_analysis,
        )
    
    # ============================================
    # Process Analysis
    # ============================================
    
    if st.session_state.is_processing and user_query:
        with st.spinner("🔄 Analyzing your data..."):
            with st.expander("📋 Analysis Steps", expanded=True):
                try:
                    # Run analysis
                    result = st.session_state.loop.run_until_complete(
                        run_analysis(st.session_state.file_path, user_query)
                    )
                    
                    st.session_state.last_result = {
                        'query': user_query,
                        'response': result['response'],
                        'chart': result.get('chart'),
                        'analysis': result.get('analysis')
                    }
                    
                    # Add to history
                    st.session_state.analysis_history.append(st.session_state.last_result)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.last_result = None
        
        st.session_state.is_processing = False
        st.rerun()
    
    # ============================================
    # Display Results
    # ============================================
    
    if st.session_state.last_result:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Display query
        st.markdown(f"**Your Question:** {st.session_state.last_result['query']}")
        
        # Display response
        with st.container():
            st.markdown("### 💡 Insights")
            st.markdown(st.session_state.last_result['response'])
        
        # Display chart if available
        if st.session_state.last_result.get('chart'):
            st.markdown("### 📈 Visualization")
            chart_data = base64.b64decode(st.session_state.last_result['chart'])
            st.image(chart_data, use_container_width=True)
        
        # Download options
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download text response
            st.download_button(
                label="📄 Download Analysis (TXT)",
                data=f"Question: {st.session_state.last_result['query']}\n\n{st.session_state.last_result['response']}",
                file_name="analysis_results.txt",
                mime="text/plain"
            )
        
        with col2:
            # Download chart if available
            if st.session_state.last_result.get('chart'):
                chart_data = base64.b64decode(st.session_state.last_result['chart'])
                st.download_button(
                    label="📊 Download Chart (PNG)",
                    data=chart_data,
                    file_name="analysis_chart.png",
                    mime="image/png"
                )
        
        # Ask another question
        st.markdown("---")
        if st.button("🔄 Ask Another Question", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()

else:
    # Help section when no file uploaded
    st.markdown("---")
    st.markdown("""
        <div style='padding: 30px; background-color: #f0f2f6; border-radius: 15px;'>
        <h3>🚀 Getting Started</h3>
        <ol>
            <li><strong>Enter your OpenAI API Key</strong> in the sidebar (or set it in .env file)</li>
            <li><strong>Upload your data file</strong> (CSV or Excel format)</li>
            <li><strong>Ask questions</strong> about your data in natural language</li>
            <li><strong>Get instant insights</strong> with automatic visualizations</li>
            <li><strong>Download results</strong> for your records</li>
        </ol>
        
        <h3 style='margin-top: 20px;'>✨ Features</h3>
        <ul>
            <li>🤖 Natural language queries - no coding required</li>
            <li>📊 Automatic chart generation when helpful</li>
            <li>💡 AI-powered insights and analysis</li>
            <li>📥 Download results and visualizations</li>
            <li>🔒 Secure - your data stays local</li>
        </ul>
        
        <h3 style='margin-top: 20px;'>💼 Use Cases</h3>
        <ul>
            <li>Sales performance analysis</li>
            <li>Customer behavior insights</li>
            <li>Financial data exploration</li>
            <li>Marketing campaign analytics</li>
            <li>Operational metrics review</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

# ============================================
# Analysis History
# ============================================

if len(st.session_state.analysis_history) > 1:
    st.markdown("---")
    st.markdown("## 📜 Analysis History")
    
    with st.expander(f"View previous analyses ({len(st.session_state.analysis_history) - 1})"):
        for idx, item in enumerate(reversed(st.session_state.analysis_history[:-1])):
            st.markdown(f"**Question {len(st.session_state.analysis_history) - idx - 1}:** {item['query']}")
            st.markdown(item['response'][:200] + "..." if len(item['response']) > 200 else item['response'])
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
    Built with ❤️ using LangGraph, GPT-4o, and Streamlit<br>
    Made by Atul for data-driven insights
    </div>
""", unsafe_allow_html=True)