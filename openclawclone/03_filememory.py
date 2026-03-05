# ---------------------------------------------------------------------------
# 03_filememory.py — Persistent memory with AsyncSqliteSaver.
#
# One change from 02: swap InMemorySaver for AsyncSqliteSaver.
# Conversation history now survives bot restarts — stored in sessions.db.
#
# Why post_init? AsyncSqliteSaver needs an async SQLite connection, which
# can only be opened inside an async context. post_init is a python-telegram-
# bot hook called after the Application is fully initialised — the right
# place to set up async resources before polling begins.
#
# Run: python 03_filememory.py
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

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))

# graph is built async in post_init — cannot be created at import time
graph = None

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ---------------------------------------------------------------------------
# ASYNC GRAPH SETUP
#
# Called once by python-telegram-bot after the Application is ready.
# Opens an async SQLite connection and compiles the graph with it.
# sessions.db is created automatically if it doesn't exist.
# ---------------------------------------------------------------------------

async def post_init(application):
    global graph
    conn = await aiosqlite.connect("sessions.db")

    # AsyncSqliteSaver: persists checkpoints to sessions.db on disk.
    # Replaces InMemorySaver — memory now survives restarts.
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

# post_init is passed here — telegram-bot calls it before run_polling()
app = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).post_init(post_init).build()
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
