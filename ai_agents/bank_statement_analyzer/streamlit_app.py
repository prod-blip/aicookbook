"""
Bank Statement Analyzer - Streamlit Frontend

Upload PDF bank statements and get AI-powered spending analysis with
automatic categorization, recurring expense detection, and visual insights.
"""

import os
import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

from agent import run_analyzer

# Load environment variables
load_dotenv()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Bank Statement Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stDownloadButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)

    if "workflow_stage" not in st.session_state:
        st.session_state.workflow_stage = "upload"  # or "results"

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None

init_session_state()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 🔑 Configuration Status")

    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    st.markdown(f"{'✅' if has_openai_key else '❌'} OpenAI API Key")

    if not has_openai_key:
        st.warning("⚠️ Please set OPENAI_API_KEY in .env file")

    st.markdown("---")
    st.markdown("### 📋 What This Does")
    st.markdown("""
    1. **Upload** PDF bank statement
    2. **Extract** transactions automatically
    3. **Categorize** spending with AI
    4. **Analyze** patterns and trends
    5. **Visualize** insights with charts
    """)

    st.markdown("---")
    st.markdown("### ℹ️ Features")
    st.markdown("""
    - Automatic transaction extraction
    - Smart AI categorization
    - Recurring expense detection
    - Category-wise breakdown
    - Daily spending trends
    - Top merchant analysis
    """)

    st.markdown("---")
    st.markdown("### 🏷️ Categories")
    st.markdown("""
    - Food & Dining
    - Transport
    - Shopping
    - Utilities
    - Bills & Recharges
    - Entertainment
    - Healthcare
    - Transfers
    - Miscellaneous Payments
    - Others
    """)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_report(results):
    """Generate text report for download"""
    report = []
    report.append("="*60)
    report.append("BANK STATEMENT ANALYSIS REPORT")
    report.append("="*60)
    report.append("")

    # Metadata
    if results.get("parsing_metadata"):
        metadata = results["parsing_metadata"]
        report.append(f"Account Number: {metadata.get('account_number', 'N/A')}")
        report.append(f"Statement Period: {metadata.get('statement_period', 'N/A')}")
        report.append("")

    # Summary
    report.append("SUMMARY")
    report.append("-"*60)
    report.append(f"Total Transactions: {results.get('transaction_count', 0)}")
    report.append(f"Total Debit: ₹{results.get('total_debit', 0):,.2f}")
    report.append(f"Total Credit: ₹{results.get('total_credit', 0):,.2f}")
    report.append(f"Net Balance Change: ₹{results.get('net_balance_change', 0):,.2f}")
    report.append("")

    # Category Breakdown
    report.append("SPENDING BY CATEGORY")
    report.append("-"*60)
    if results.get("category_totals"):
        for cat, amount in sorted(results["category_totals"].items(), key=lambda x: x[1], reverse=True):
            report.append(f"{cat:30s}: ₹{amount:,.2f}")
    report.append("")

    # Recurring Expenses
    report.append("RECURRING EXPENSES")
    report.append("-"*60)
    if results.get("recurring_expenses"):
        for rec in results["recurring_expenses"]:
            report.append(f"{rec['merchant']}")
            report.append(f"  Category: {rec['category']}")
            report.append(f"  Frequency: {rec['frequency']}")
            report.append(f"  Avg Amount: ₹{rec['avg_amount']:,.2f}")
            report.append(f"  Total: ₹{rec['total']:,.2f}")
            report.append("")
    else:
        report.append("No recurring expenses detected.")
        report.append("")

    # Top Merchants
    report.append("TOP MERCHANTS")
    report.append("-"*60)
    if results.get("top_merchants"):
        for i, merch in enumerate(results["top_merchants"], 1):
            report.append(f"{i}. {merch['merchant']:40s} ₹{merch['total']:,.2f} ({merch['count']} txns)")
    report.append("")

    report.append("="*60)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*60)

    return "\n".join(report)


def reset_to_upload():
    """Reset to upload stage"""
    st.session_state.workflow_stage = "upload"
    st.session_state.analysis_results = None
    st.session_state.uploaded_file_path = None
    st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<div class="main-header">💰 Bank Statement Analyzer</div>', unsafe_allow_html=True)

# ============================================================================
# STAGE 1: UPLOAD
# ============================================================================

if st.session_state.workflow_stage == "upload":
    st.markdown("### 📤 Upload Your Bank Statement")

    # Help text
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Step 1:** Upload your PDF bank statement using the file uploader below.

        **Step 2:** Click "Analyze Statement" to start the AI-powered analysis.

        **Step 3:** View comprehensive insights including:
        - Category-wise spending breakdown
        - Recurring expense detection
        - Daily spending trends
        - Top merchants analysis

        **Supported Format:** PDF bank statements (digital, not scanned)

        **Privacy:** All processing happens securely. Your data is not stored.
        """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload your bank statement in PDF format"
    )

    if uploaded_file:
        # Save to temp location
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_path = temp_path
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💰 Analyze Statement", type="primary", use_container_width=True):
                if not has_openai_key:
                    st.error("❌ Please set OPENAI_API_KEY in .env file before analyzing.")
                else:
                    st.session_state.is_processing = True

        # Process if button clicked
        if st.session_state.is_processing:
            with st.spinner("🔄 Analyzing your statement... This may take 20-30 seconds."):
                try:
                    # Run analyzer
                    result = st.session_state.loop.run_until_complete(
                        run_analyzer(temp_path, uploaded_file.name)
                    )

                    # Check for errors
                    if result.get("errors"):
                        st.error(f"❌ Analysis failed: {', '.join(result['errors'])}")
                        st.session_state.is_processing = False
                    else:
                        st.session_state.analysis_results = result
                        st.session_state.workflow_stage = "results"
                        st.session_state.is_processing = False
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.is_processing = False

    else:
        st.info("👆 Please upload a PDF bank statement to get started.")

# ============================================================================
# STAGE 2: RESULTS DASHBOARD
# ============================================================================

elif st.session_state.workflow_stage == "results":
    results = st.session_state.analysis_results

    # Back button
    if st.button("⬅️ Analyze Another Statement"):
        reset_to_upload()

    st.markdown("---")

    # ========================================================================
    # METADATA
    # ========================================================================

    if results.get("parsing_metadata"):
        metadata = results["parsing_metadata"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Account:** {metadata.get('account_number', 'N/A')}")
        with col2:
            st.markdown(f"**Period:** {metadata.get('statement_period', 'N/A')}")

    st.markdown("---")

    # ========================================================================
    # SUMMARY CARDS
    # ========================================================================

    st.markdown("### 📊 Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Spent",
            value=f"₹{results.get('total_debit', 0):,.2f}",
            delta=None
        )

    with col2:
        st.metric(
            label="Total Received",
            value=f"₹{results.get('total_credit', 0):,.2f}",
            delta=None
        )

    with col3:
        if results.get("category_totals"):
            top_cat = max(results['category_totals'].items(), key=lambda x: x[1])
            st.metric(
                label="Top Category",
                value=top_cat[0],
                delta=f"₹{top_cat[1]:,.2f}"
            )
        else:
            st.metric(label="Top Category", value="N/A")

    with col4:
        st.metric(
            label="Recurring Expenses",
            value=len(results.get('recurring_expenses', [])),
            delta=None
        )

    st.markdown("---")

    # ========================================================================
    # CHARTS
    # ========================================================================

    st.markdown("### 📈 Visual Insights")

    col_left, col_right = st.columns(2)

    # Pie Chart - Category Breakdown
    with col_left:
        st.markdown("#### 🥧 Spending by Category")

        if results.get("category_totals"):
            fig_pie = px.pie(
                values=list(results['category_totals'].values()),
                names=list(results['category_totals'].keys()),
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>₹%{value:,.2f}<br>%{percent}<extra></extra>'
            )
            fig_pie.update_layout(
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No category data available")

    # Line Chart - Daily Spending Trend
    with col_right:
        st.markdown("#### 📉 Daily Spending Trend")

        if results.get("daily_spending"):
            # Sort by date
            daily_data = sorted(results['daily_spending'].items())
            dates = [item[0] for item in daily_data]
            amounts = [item[1] for item in daily_data]

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=dates,
                y=amounts,
                mode='lines+markers',
                name='Daily Spending',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{x}</b><br>₹%{y:,.2f}<extra></extra>'
            ))
            fig_line.update_layout(
                xaxis_title="Date",
                yaxis_title="Amount (₹)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No daily spending data available")

    st.markdown("---")

    # ========================================================================
    # TOP MERCHANTS BAR CHART
    # ========================================================================

    st.markdown("### 🏪 Top Merchants")

    if results.get("top_merchants"):
        merchants = [m['merchant'][:30] for m in results['top_merchants']]  # Truncate long names
        totals = [m['total'] for m in results['top_merchants']]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=totals,
            y=merchants,
            orientation='h',
            marker=dict(
                color=totals,
                colorscale='Viridis',
                showscale=True
            ),
            hovertemplate='<b>%{y}</b><br>₹%{x:,.2f}<extra></extra>'
        ))
        fig_bar.update_layout(
            xaxis_title="Total Spent (₹)",
            yaxis_title="Merchant",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No merchant data available")

    st.markdown("---")

    # ========================================================================
    # TRANSACTION TABLE
    # ========================================================================

    st.markdown("### 📋 All Transactions")

    if results.get("categorized_transactions"):
        df_txn = pd.DataFrame(results['categorized_transactions'])

        # Reorder columns
        column_order = ['date', 'description', 'amount', 'type', 'category']
        df_txn = df_txn[column_order]

        # Rename columns for display
        df_txn.columns = ['Date', 'Description', 'Amount (₹)', 'Type', 'Category']

        # Add filter by category
        categories = ['All'] + sorted(df_txn['Category'].unique().tolist())
        selected_category = st.selectbox("Filter by Category:", categories)

        if selected_category != 'All':
            df_filtered = df_txn[df_txn['Category'] == selected_category]
        else:
            df_filtered = df_txn

        st.dataframe(
            df_filtered,
            use_container_width=True,
            height=400
        )

        st.caption(f"Showing {len(df_filtered)} of {len(df_txn)} transactions")
    else:
        st.info("No transactions available")

    st.markdown("---")

    # ========================================================================
    # RECURRING EXPENSES
    # ========================================================================

    st.markdown("### 🔁 Recurring Expenses")

    if results.get("recurring_expenses") and len(results['recurring_expenses']) > 0:
        with st.expander(f"📌 Found {len(results['recurring_expenses'])} recurring expenses", expanded=True):
            for i, rec in enumerate(results['recurring_expenses'], 1):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                with col1:
                    st.markdown(f"**{i}. {rec['merchant']}**")
                    st.caption(f"Category: {rec['category']}")

                with col2:
                    st.markdown(f"**{rec['frequency']}**")
                    st.caption("Frequency")

                with col3:
                    st.markdown(f"**₹{rec['avg_amount']:,.2f}**")
                    st.caption("Avg Amount")

                with col4:
                    st.markdown(f"**₹{rec['total']:,.2f}**")
                    st.caption("Total")

                if i < len(results['recurring_expenses']):
                    st.markdown("---")
    else:
        st.info("No recurring expenses detected in this statement.")

    st.markdown("---")

    # ========================================================================
    # DOWNLOAD REPORT
    # ========================================================================

    st.markdown("### 📥 Download Report")

    report_text = generate_report(results)

    st.download_button(
        label="📄 Download Analysis Report (TXT)",
        data=report_text,
        file_name=f"statement_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Tech Stack:</strong> LangGraph • GPT-4o • Streamlit • Plotly</p>
    <p>Built with ❤️ for smarter financial insights</p>
</div>
""", unsafe_allow_html=True)
