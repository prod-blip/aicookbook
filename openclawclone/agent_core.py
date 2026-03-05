"""
agent_core.py — Channel-agnostic agent logic.

This module owns:
  - Loading SOUL.md + TOOLS.md into the system prompt
  - Building the LangGraph graph (with SQLite memory + tool node)
  - run_turn(): the single function any channel calls to talk to the agent

No Telegram, no HTTP, no channel-specific code lives here.
"""

import os
import aiosqlite
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import tools_condition

from tools import tools, tool_node, set_current_thread_id

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
#
# SOUL.md defines personality. TOOLS.md describes available tools.
# Both are injected on every LLM call so the agent is always grounded.
# ---------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r") as f:
    SOUL = f.read()

with open(os.path.join(os.path.dirname(__file__), "TOOLS.md"), "r") as f:
    TOOLS_DOC = f.read()

SYSTEM_PROMPT = SOUL + "\n\n" + TOOLS_DOC

print("✅ SOUL and TOOLS loaded")

# ---------------------------------------------------------------------------
# LLM
#
# bind_tools() registers the tool schemas with the model so it knows when
# and how to call each tool.
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

# Graph is initialised async in setup_graph() — not at import time.
graph = None


# ---------------------------------------------------------------------------
# LANGGRAPH NODE
# ---------------------------------------------------------------------------

def call_model(state: MessagesState):
    """
    Core LLM node. Prepends the system prompt on every call.
    Uses llm_with_tools so the model can invoke tools when needed.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# GRAPH SETUP
#
# Called once at startup (from the gateway's post_init hook).
# Opens the SQLite connection and compiles the graph.
#
# Flow:
#   START → call_model → [tool_calls?] → tools → call_model → END
#                      → [no tool_calls] → END
# ---------------------------------------------------------------------------

async def setup_graph():
    """Initialise the LangGraph graph with SQLite checkpointing."""
    global graph
    conn = await aiosqlite.connect(
        os.path.join(os.path.dirname(__file__), "sessions.db")
    )
    checkpointer = AsyncSqliteSaver(conn)
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    graph = builder.compile(checkpointer=checkpointer)
    print("✅ Agent graph initialized")


# ---------------------------------------------------------------------------
# RUN TURN — the single entry point for all channels
#
# Any channel (Telegram, HTTP, Discord, etc.) calls this with:
#   - thread_id    : prefixed by channel, e.g. "telegram_123" or "http_456"
#   - user_message : the raw text from the user
#
# Returns the agent's reply as a plain string.
# ---------------------------------------------------------------------------

async def run_turn(thread_id: str, user_message: str) -> str:
    """
    Invoke the agent for one user message.
    Channel-agnostic: just pass a thread_id and message, get a string back.
    """
    # Make thread_id available to tools (for pending_approvals keying)
    set_current_thread_id(thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
    )
    return response["messages"][-1].content
