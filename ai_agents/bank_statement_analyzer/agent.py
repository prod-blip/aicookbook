"""
Bank Statement Analyzer Agent

A LangGraph-based agent that extracts transactions from PDF bank statements,
categorizes spending using LLM, and provides comprehensive financial analytics.

Architecture:
    load_pdf -> parse_transactions -> categorize_transactions -> analyze_spending -> prepare_visualizations
"""

import os
import json
from typing import TypedDict, Annotated, Optional, List, Dict
from collections import defaultdict
import operator

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END

# ============================================================================
# CONSTANTS
# ============================================================================

LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0  # Deterministic parsing/categorization

PREDEFINED_CATEGORIES = [
    "Food & Dining",
    "Transport",
    "Shopping",
    "Utilities",
    "Bills & Recharges",
    "Entertainment",
    "Healthcare",
    "Transfers",
    "Miscellaneous Payments",  # Person-to-person UPI transfers
    "Others"
]

MIN_RECURRING_COUNT = 2
TOP_MERCHANTS_LIMIT = 10

# ============================================================================
# STATE SCHEMA
# ============================================================================

class BankStatementState(TypedDict):
    """State for Bank Statement Analyzer workflow"""

    # Input
    pdf_path: str
    filename: str

    # Processing Outputs
    raw_text: Optional[str]
    page_count: Optional[int]
    transactions: Optional[List[Dict]]  # Raw parsed transactions
    transaction_count: Optional[int]
    parsing_metadata: Optional[Dict]  # Account number, statement period

    # Categorization
    categorized_transactions: Optional[List[Dict]]
    # Format: [{"date": "DD-MM-YYYY", "description": str, "amount": float, "type": "debit/credit", "category": str}]

    # Analysis Results
    category_totals: Optional[Dict[str, float]]
    total_debit: Optional[float]
    total_credit: Optional[float]
    net_balance_change: Optional[float]
    recurring_expenses: Optional[List[Dict]]
    daily_spending: Optional[Dict[str, float]]
    top_merchants: Optional[List[Dict]]

    # Visualization
    chart_data: Optional[Dict]

    # Error Tracking
    messages: Annotated[list, operator.add]
    errors: Annotated[List[str], operator.add]


# ============================================================================
# NODE 1: LOAD PDF
# ============================================================================

async def load_pdf(state: BankStatementState) -> BankStatementState:
    """
    Load and extract text from PDF bank statement.

    Reads: state["pdf_path"]
    Updates: state["raw_text"], state["page_count"]
    """
    print("\n=== 📄 Loading PDF ===")

    try:
        pdf_path = state["pdf_path"]
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Combine all pages (bank statements are usually 1-3 pages)
        raw_text = "\n".join([page.page_content for page in pages])
        page_count = len(pages)

        print(f"✅ Loaded {page_count} pages, {len(raw_text)} characters")

        return {
            **state,
            "raw_text": raw_text,
            "page_count": page_count,
        }

    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return {
            **state,
            "raw_text": None,
            "page_count": 0,
            "errors": [f"PDF loading failed: {str(e)}"]
        }


# ============================================================================
# NODE 2: PARSE TRANSACTIONS
# ============================================================================

async def parse_transactions(state: BankStatementState) -> BankStatementState:
    """
    Parse raw text into structured transaction list using LLM.

    Reads: state["raw_text"], state["filename"]
    Updates: state["transactions"], state["transaction_count"], state["parsing_metadata"]
    """
    print("\n=== 🔍 Parsing Transactions ===")

    if not state["raw_text"]:
        print("⏭️ No text to parse, skipping")
        return {**state}

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=LLM_TEMPERATURE
        )

        prompt = f"""
You are a bank statement parser. Extract ALL transactions from this statement text.

Statement Text:
{state['raw_text']}

Return ONLY a JSON object with this EXACT structure:
{{
    "account_number": "masked account number (e.g., XXXX1234)",
    "statement_period": "date range",
    "transactions": [
        {{
            "date": "DD-MM-YYYY",
            "description": "merchant/transaction description",
            "amount": 1234.56,
            "type": "debit" or "credit"
        }},
        ...
    ]
}}

Rules:
- Extract EVERY transaction from the statement
- date: Parse from statement format to DD-MM-YYYY
- description: Clean merchant name (remove extra codes/spaces)
- amount: Always positive float
- type: "debit" for money out, "credit" for money in
- Return ONLY the JSON object, no markdown, no code blocks
"""

        response = await llm.ainvoke(prompt)
        result_text = response.content.strip()

        # Clean markdown if present
        if result_text.startswith("```"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        parsed_data = json.loads(result_text)

        transactions = parsed_data.get("transactions", [])
        transaction_count = len(transactions)

        metadata = {
            "account_number": parsed_data.get("account_number", "Unknown"),
            "statement_period": parsed_data.get("statement_period", "Unknown")
        }

        print(f"✅ Parsed {transaction_count} transactions")

        return {
            **state,
            "transactions": transactions,
            "transaction_count": transaction_count,
            "parsing_metadata": metadata,
        }

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return {
            **state,
            "transactions": None,
            "transaction_count": 0,
            "errors": [f"Transaction parsing failed: Invalid JSON from LLM"]
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            **state,
            "transactions": None,
            "transaction_count": 0,
            "errors": [f"Transaction parsing failed: {str(e)}"]
        }


# ============================================================================
# NODE 3: CATEGORIZE TRANSACTIONS
# ============================================================================

async def categorize_transactions(state: BankStatementState) -> BankStatementState:
    """
    Categorize all transactions using LLM.

    Reads: state["transactions"]
    Updates: state["categorized_transactions"]
    """
    print("\n=== 🏷️ Categorizing Transactions ===")

    if not state["transactions"]:
        print("⏭️ No transactions to categorize, skipping")
        return {**state}

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=LLM_TEMPERATURE
        )

        # Batch all transactions in one LLM call
        transactions_json = json.dumps(state["transactions"], indent=2)

        prompt = f"""
You are a transaction categorizer specialized in Indian bank transactions. Assign each transaction to ONE category.

CATEGORIES (use EXACTLY these):
- Food & Dining (restaurants, cafes, food delivery, groceries, juice shops)
- Transport (fuel, Uber, Rapido, metro, parking, tolls, cabs)
- Shopping (clothing, electronics, retail, e-commerce, Bata, Amazon)
- Utilities (electricity, water, gas, internet, phone bills, broadband)
- Bills & Recharges (mobile recharge, DTH, subscriptions, insurance)
- Entertainment (movies, concerts, gaming, streaming, OTT)
- Healthcare (pharmacy, doctor, hospital, medical, Urban Company health)
- Transfers (bank transfers, loan payments, credit card payments)
- Miscellaneous Payments (person-to-person UPI transfers to individuals)
- Others (anything that doesn't fit above)

SPECIAL INSTRUCTIONS FOR UPI TRANSACTIONS:

Many transactions follow this format: UPI/[MERCHANT]/[APP]/[PURPOSE]/[BANK]

Rules for UPI Analysis:
1. **Extract merchant name** from UPI format (text between first two slashes)
   Example: "UPI/Rapido/paytm-123/Payment" → Merchant = "Rapido"

2. **Identify transaction type:**
   - PERSON: Contains person name + phone number → "Miscellaneous Payments"
   - BUSINESS: Company/service name → Categorize by service type

3. **P2P (Person-to-Person) Detection:**
   - Look for: Name + Phone pattern (e.g., "HARSHITA SHARMA/9876543210@paytm")
   - All P2P → "Miscellaneous Payments"

4. **Business Transaction Intelligence:**
   - Extract actual merchant/service name (ignore UPI codes)
   - Categorize based on what the business does, not transaction format

   Examples:
   - "MUMBAI MET"/"MUMBAIMETROPG" → Mumbai Metro service → "Transport"
   - "Rapido" → Ride hailing app → "Transport"
   - "Zomato"/"Swiggy" → Food delivery → "Food & Dining"
   - "Vodafone"/"VI" → Telecom → "Utilities"
   - "Urban Company" → Check context (salon/cleaning/repair) → Appropriate category

5. **Use merchant context, NOT just keywords:**
   - Focus on what service the merchant provides
   - Ignore payment app names (Paytm, PhonePe, GPay)
   - Ignore bank names in description

EXAMPLES:

Input: "UPI/HARSHITA SHARMA/9876543210@paytm/Payment from/HDFC"
Analysis: Person name + phone number detected
Output: "Miscellaneous Payments"

Input: "UPI/Rapido/paytm-76881028/Payment fr/YES BANK"
Analysis: Rapido = ride hailing service
Output: "Transport"

Input: "UPI/MUMBAI MET/MUMBAIMETROPG@/Payment fo/YES"
Analysis: Mumbai Metropolitan = Mumbai Metro service
Output: "Transport"

Input: "UPI/Zomato/zomato@paytm/Payment from/HDFC"
Analysis: Zomato = food delivery service
Output: "Food & Dining"

Input: "UPI/JOHN DOE/john@paytm/Transfer"
Analysis: Person name detected
Output: "Miscellaneous Payments"

Transactions:
{transactions_json}

Return ONLY a JSON array with this structure:
[
    {{
        "date": "DD-MM-YYYY",
        "description": "original description",
        "amount": 1234.56,
        "type": "debit",
        "category": "Food & Dining"
    }},
    ...
]

Rules:
- Keep date, description, amount, type EXACTLY as provided
- Add category field based on intelligent analysis
- Use ONLY the categories listed above
- Return same number of items as input
- Return ONLY the JSON array, no markdown, no code blocks
"""

        response = await llm.ainvoke(prompt)
        result_text = response.content.strip()

        # Clean markdown
        if result_text.startswith("```"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()

        categorized_transactions = json.loads(result_text)

        print(f"✅ Categorized {len(categorized_transactions)} transactions")

        return {
            **state,
            "categorized_transactions": categorized_transactions,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            **state,
            "categorized_transactions": state["transactions"],  # Fallback to uncategorized
            "errors": [f"Categorization failed: {str(e)}"]
        }


# ============================================================================
# NODE 4: ANALYZE SPENDING
# ============================================================================

async def analyze_spending(state: BankStatementState) -> BankStatementState:
    """
    Calculate spending analytics and detect recurring expenses.

    Reads: state["categorized_transactions"]
    Updates: state["category_totals"], state["total_debit"], state["total_credit"],
             state["net_balance_change"], state["recurring_expenses"],
             state["daily_spending"], state["top_merchants"]
    """
    print("\n=== 📊 Analyzing Spending ===")

    if not state["categorized_transactions"]:
        print("⏭️ No transactions to analyze, skipping")
        return {**state}

    try:
        transactions = state["categorized_transactions"]

        # 1. Category-wise totals
        category_totals = defaultdict(float)
        for txn in transactions:
            if txn["type"] == "debit":
                category_totals[txn["category"]] += txn["amount"]

        # 2. Overall totals
        total_debit = sum(t["amount"] for t in transactions if t["type"] == "debit")
        total_credit = sum(t["amount"] for t in transactions if t["type"] == "credit")
        net_balance_change = total_credit - total_debit

        # 3. Daily spending trend
        daily_spending = defaultdict(float)
        for txn in transactions:
            if txn["type"] == "debit":
                daily_spending[txn["date"]] += txn["amount"]

        # 4. Top merchants
        merchant_totals = defaultdict(lambda: {"total": 0, "count": 0})
        for txn in transactions:
            if txn["type"] == "debit":
                merchant = txn["description"][:50]  # Truncate long names
                merchant_totals[merchant]["total"] += txn["amount"]
                merchant_totals[merchant]["count"] += 1

        top_merchants = [
            {"merchant": merch, "total": data["total"], "count": data["count"]}
            for merch, data in sorted(merchant_totals.items(),
                                     key=lambda x: x[1]["total"],
                                     reverse=True)[:TOP_MERCHANTS_LIMIT]
        ]

        # 5. Detect recurring expenses (same merchant, multiple occurrences)
        merchant_transactions = defaultdict(list)
        for txn in transactions:
            if txn["type"] == "debit":
                merchant_transactions[txn["description"]].append(txn)

        recurring_expenses = []
        for merchant, txns in merchant_transactions.items():
            if len(txns) >= MIN_RECURRING_COUNT:  # Appears at least twice
                avg_amount = sum(t["amount"] for t in txns) / len(txns)
                recurring_expenses.append({
                    "merchant": merchant,
                    "category": txns[0]["category"],
                    "frequency": f"{len(txns)}x in statement period",
                    "avg_amount": round(avg_amount, 2),
                    "total": round(sum(t["amount"] for t in txns), 2),
                    "transactions": txns
                })

        # Sort by total amount
        recurring_expenses.sort(key=lambda x: x["total"], reverse=True)

        print(f"✅ Analysis complete:")
        print(f"   - {len(category_totals)} categories")
        print(f"   - {len(recurring_expenses)} recurring expenses")
        print(f"   - Net balance change: ₹{net_balance_change:,.2f}")

        return {
            **state,
            "category_totals": dict(category_totals),
            "total_debit": round(total_debit, 2),
            "total_credit": round(total_credit, 2),
            "net_balance_change": round(net_balance_change, 2),
            "recurring_expenses": recurring_expenses,
            "daily_spending": dict(daily_spending),
            "top_merchants": top_merchants,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            **state,
            "errors": [f"Analysis failed: {str(e)}"]
        }


# ============================================================================
# NODE 5: PREPARE VISUALIZATIONS
# ============================================================================

async def prepare_visualizations(state: BankStatementState) -> BankStatementState:
    """
    Structure data for Streamlit charts.

    Reads: state["category_totals"], state["daily_spending"], state["top_merchants"]
    Updates: state["chart_data"]
    """
    print("\n=== 📈 Preparing Visualizations ===")

    if not state.get("category_totals"):
        print("⏭️ No data for visualization, skipping")
        return {**state}

    try:
        chart_data = {
            "category_breakdown": state["category_totals"],
            "daily_trend": state["daily_spending"],
            "top_merchants": state["top_merchants"],
        }

        print("✅ Visualization data prepared")

        return {
            **state,
            "chart_data": chart_data,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            **state,
            "errors": [f"Visualization prep failed: {str(e)}"]
        }


# ============================================================================
# GRAPH CREATION
# ============================================================================

def create_graph():
    """Create the LangGraph workflow"""

    workflow = StateGraph(BankStatementState)

    # Add nodes
    workflow.add_node("load_pdf", load_pdf)
    workflow.add_node("parse_transactions", parse_transactions)
    workflow.add_node("categorize_transactions", categorize_transactions)
    workflow.add_node("analyze_spending", analyze_spending)
    workflow.add_node("prepare_visualizations", prepare_visualizations)

    # Set entry point
    workflow.set_entry_point("load_pdf")

    # Add edges (sequential flow)
    workflow.add_edge("load_pdf", "parse_transactions")
    workflow.add_edge("parse_transactions", "categorize_transactions")
    workflow.add_edge("categorize_transactions", "analyze_spending")
    workflow.add_edge("analyze_spending", "prepare_visualizations")
    workflow.add_edge("prepare_visualizations", END)

    return workflow.compile()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

async def run_analyzer(pdf_path: str, filename: str) -> Dict:
    """
    Main function to run the bank statement analyzer

    Args:
        pdf_path: Path to the PDF file
        filename: Original filename

    Returns:
        Dict with all analysis results
    """
    print("\n" + "="*60)
    print("🏦 BANK STATEMENT ANALYZER")
    print("="*60)

    # Create graph
    graph = create_graph()

    # Initial state
    initial_state = {
        "pdf_path": pdf_path,
        "filename": filename,
        "raw_text": None,
        "page_count": None,
        "transactions": None,
        "transaction_count": None,
        "parsing_metadata": None,
        "categorized_transactions": None,
        "category_totals": None,
        "total_debit": None,
        "total_credit": None,
        "net_balance_change": None,
        "recurring_expenses": None,
        "daily_spending": None,
        "top_merchants": None,
        "chart_data": None,
        "messages": [],
        "errors": []
    }

    # Run the graph
    result = await graph.ainvoke(initial_state)

    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)

    return result


# ============================================================================
# TESTING (if run directly)
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    # Test with a sample PDF
    test_pdf_path = "sample_statement.pdf"

    async def test():
        result = await run_analyzer(test_pdf_path, "sample_statement.pdf")
        print(f"\nTransactions found: {result.get('transaction_count')}")
        print(f"Categories: {list(result.get('category_totals', {}).keys())}")
        print(f"Errors: {result.get('errors')}")

    asyncio.run(test())
