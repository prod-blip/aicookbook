# ---------------------------------------------------------------------------
# 04_memorywithsoul.py — Persistent memory + personality via SOUL.md.
#
# Adds a system prompt so the agent has a consistent name, tone, and
# behaviour. The personality is defined in SOUL.md — a plain markdown file
# that can be edited without touching any Python code.
#
# Key pattern: SOUL is loaded once at startup and prepended to state["messages"]
# on every call_model invocation. It is never stored in the checkpoint —
# editing SOUL.md takes effect on the next message, no restart needed.
#
# Run: python 04_memorywithsoul.py
# ---------------------------------------------------------------------------

import os
import aiosqlite
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, MessagesState, START
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

print("✅ Token loaded:", bool(os.getenv("TELEGRAM_BOT_TOKEN")))

# ---------------------------------------------------------------------------
# SOUL — agent personality loaded from SOUL.md.
# Contains name, tone, boundaries, and memory instructions.
# Injected as a system message on every LLM call.
# ---------------------------------------------------------------------------

with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r") as f:
    SOUL = f.read()

print("✅ SOUL loaded from SOUL.md")

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))

graph = None

# ---------------------------------------------------------------------------
# LANGGRAPH NODE
#
# SOUL is prepended as a system message before every LLM call.
# This ensures personality is always in context, regardless of how many
# messages are in history.
# ---------------------------------------------------------------------------

def call_model(state: MessagesState):
    # System message always comes first — never stored in checkpoint
    messages = [{"role": "system", "content": SOUL}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

async def post_init(application):
    global graph
    conn = await aiosqlite.connect("sessions.db")
    checkpointer = AsyncSqliteSaver(conn)
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    graph = builder.compile(checkpointer=checkpointer)
    print("✅ Graph with SQLite memory initialized")

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
