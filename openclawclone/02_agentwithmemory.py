# ---------------------------------------------------------------------------
# 02_agentwithmemory.py — Conversation memory with LangGraph + InMemorySaver.
#
# Introduces LangGraph's StateGraph so the agent accumulates message history
# within a session. Each user gets an isolated history via their thread_id
# (Telegram chat ID).
#
# Limitation: InMemorySaver stores checkpoints in RAM only.
# Memory is lost when the bot restarts — see 03_filememory.py for the fix.
#
# Run: python 02_agentwithmemory.py
# ---------------------------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

print("✅ Token loaded:", bool(os.getenv("TELEGRAM_BOT_TOKEN")))

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))

# InMemorySaver: stores all checkpoints in RAM.
# Fast and zero-setup, but lost on every restart.
checkpointer = InMemorySaver()

# ---------------------------------------------------------------------------
# LANGGRAPH NODE
# call_model receives the full message history via state["messages"] and
# appends the LLM's reply. LangGraph's add_messages reducer handles merging.
# ---------------------------------------------------------------------------

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ---------------------------------------------------------------------------
# GRAPH
# Simple single-node graph: START → call_model → END
# The checkpointer persists state between calls using thread_id as the key.
# ---------------------------------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)

async def handle_message(update: Update, context):
    user_message = update.message.text

    # thread_id = Telegram chat ID — each user gets their own isolated history
    thread_id = str(update.effective_chat.id)

    # LangGraph loads the checkpoint for this thread_id, appends the new
    # message, runs call_model, and saves the updated checkpoint automatically
    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
    )

    await update.message.reply_text(response["messages"][-1].content)

app = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
