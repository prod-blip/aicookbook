# ---------------------------------------------------------------------------
# 05_agentwithtool.py — File tools with LangGraph's tool loop.
#
# Gives the agent the ability to act — read and write files on the Desktop.
#
# Key concepts introduced:
#   @tool        — decorator that turns a Python function into a LangChain tool.
#                  The docstring becomes the tool's description for the LLM.
#   bind_tools() — registers tool schemas with the LLM so it knows when/how
#                  to call each tool.
#   ToolNode     — a LangGraph node that executes whatever tool the LLM called.
#   tools_condition — built-in conditional: routes to "tools" if the last
#                  message has tool_calls, otherwise routes to END.
#
# Tool loop flow:
#   START → call_model → [tool calls?] → tools → call_model → END
#                      → [no tool calls] → END
#
# Also adds TOOLS.md — a markdown description of available tools appended
# to the system prompt so the agent knows what actions it can take.
#
# Run: python 05_agentwithtool.py
# ---------------------------------------------------------------------------

import os
import aiosqlite
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

print("✅ Token loaded:", bool(os.getenv("TELEGRAM_BOT_TOKEN")))

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — SOUL + TOOLS
# TOOLS.md describes available tools so the agent knows what actions it has.
# ---------------------------------------------------------------------------

with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r") as f:
    SOUL = f.read()

with open(os.path.join(os.path.dirname(__file__), "TOOLS.md"), "r") as f:
    TOOLS_DOC = f.read()

SYSTEM_PROMPT = SOUL + "\n\n" + TOOLS_DOC

print("✅ SOUL and TOOLS loaded")

# All file operations are sandboxed to the Desktop
DESKTOP = os.path.expanduser("~/Desktop")

# ---------------------------------------------------------------------------
# TOOLS
#
# The @tool decorator wraps a Python function as a LangChain tool.
# The function's docstring is sent to the LLM as the tool description —
# keep it clear and specific so the model calls the right tool.
# ---------------------------------------------------------------------------

@tool
def read_desktop_file(filename: str) -> str:
    """Read a file from the user's Desktop and return its contents."""
    path = os.path.join(DESKTOP, filename)
    if not os.path.exists(path):
        return f"File '{filename}' not found on Desktop."
    with open(path, "r") as f:
        return f.read()

@tool
def write_desktop_file(filename: str, content: str) -> str:
    """Write content to a file on the user's Desktop."""
    path = os.path.join(DESKTOP, filename)
    with open(path, "w") as f:
        f.write(content)
    return f"✅ File '{filename}' written to Desktop."

tools = [read_desktop_file, write_desktop_file]

# ToolNode wraps the tools list — it receives an AIMessage with tool_calls
# and executes the appropriate function, returning a ToolMessage with the result
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))

# bind_tools() attaches the tool schemas to the LLM.
# This tells the model the tool names, parameters, and descriptions
# so it can decide when and how to call them.
llm_with_tools = llm.bind_tools(tools)

graph = None

# ---------------------------------------------------------------------------
# LANGGRAPH NODE
# ---------------------------------------------------------------------------

def call_model(state: MessagesState):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    # llm_with_tools — model can now emit tool_calls in its response
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ---------------------------------------------------------------------------
# GRAPH SETUP
#
# Two nodes: call_model and tools.
# tools_condition checks the last message for tool_calls:
#   - tool_calls present → route to "tools" node
#   - no tool_calls      → route to END
# After tools execute, control returns to call_model to process the result.
# ---------------------------------------------------------------------------

async def post_init(application):
    global graph
    conn = await aiosqlite.connect("sessions.db")
    checkpointer = AsyncSqliteSaver(conn)
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")  # tool result flows back to model
    graph = builder.compile(checkpointer=checkpointer)
    print("✅ Graph with tools and SQLite memory initialized")

async def handle_message(update: Update, context):
    user_message = update.message.text
    thread_id = str(update.effective_chat.id)

    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
    )

    await update.message.reply_text(response["messages"][-1].content)

app = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).post_init(post_init).build()
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
