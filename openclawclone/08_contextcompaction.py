"""
08_contextcompaction.py — Telegram bot with LangGraph-native context compaction.

Problem: After weeks of chatting, the SQLite checkpoint accumulates thousands of
messages. LangGraph loads the full history on every call — eventually exceeding
GPT-4o's 128k context window.

Solution: Extend MessagesState with a "summary" field. After every N messages,
a summarize_conversation node summarizes the older history, stores it in the
summary field (persisted in SQLite), and deletes all but the 2 most recent
messages. The summary is incremental — each new summary builds on the previous
one, so context is never lost across multiple compaction cycles.

The summary is injected into the system prompt on every call_model invocation,
so the model always has the full conversational context — just compressed.

Graph flow:
  START → call_model → route_after_model:
                         "tools"     → tools → call_model (loop)
                         "summarize" → summarize_conversation → END
                         END         → END
"""

import os
import subprocess
import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

# Import tools and permission infrastructure from tools.py
from tools import tools, tool_node, set_current_thread_id, pending_approvals, save_approval

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r") as f:
    SOUL = f.read()

with open(os.path.join(os.path.dirname(__file__), "TOOLS.md"), "r") as f:
    TOOLS_DOC = f.read()

SYSTEM_PROMPT = SOUL + "\n\n" + TOOLS_DOC

print("✅ SOUL and TOOLS loaded")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

graph = None


# ---------------------------------------------------------------------------
# STATE
#
# Extends MessagesState with a "summary" field.
#
# - "messages" is managed by LangGraph's add_messages reducer (appends/updates).
# - "summary" is a plain string we overwrite on each compaction cycle.
#
# Both fields are persisted in sessions.db, so the summary survives bot restarts.
# ---------------------------------------------------------------------------

class State(MessagesState):
    summary: str


# ---------------------------------------------------------------------------
# SUMMARIZATION THRESHOLD
#
# When message count in state reaches this value, summarize_conversation fires
# after the model replies. Keeps only the 2 most recent messages verbatim.
#
# Lower to 4 for testing (quick to trigger).
# Raise to 20–30 in production to reduce API calls.
# ---------------------------------------------------------------------------

MESSAGE_SUMMARIZE_THRESHOLD = 10


# ---------------------------------------------------------------------------
# LANGGRAPH NODES
# ---------------------------------------------------------------------------

async def call_model(state: State):
    """
    Core LLM node. Prepends the system prompt on every call.

    If a conversation summary exists in state, it is appended to the system
    prompt so the model has compressed context from earlier in the session —
    even after old messages have been deleted.

    Reads:   state["messages"], state["summary"]
    Updates: state["messages"]
    """
    summary = state.get("summary", "")
    system_content = SYSTEM_PROMPT

    # Inject prior summary into the system prompt if one exists
    if summary:
        system_content += f"\n\n[Earlier conversation — summarized]\n{summary}"

    messages = [{"role": "system", "content": system_content}] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def summarize_conversation(state: State):
    """
    Summarization node. Triggered when message count reaches MESSAGE_SUMMARIZE_THRESHOLD.

    Incremental strategy — the summary field grows to capture the full history:
      1. If a prior summary exists, extend it with the new messages.
      2. If no prior summary, create one from scratch.
      3. Delete all messages except the 2 most recent (last user turn + reply).

    Uses the base llm (not llm_with_tools) so the model cannot accidentally
    call tools during summarization instead of producing a text response.

    Reads:   state["messages"], state["summary"]
    Updates: state["summary"], state["messages"] (via RemoveMessage)
    """
    print("\n=== Summarizing conversation ===")

    summary = state.get("summary", "")

    # Build the prompt — extend prior summary if one exists, otherwise start fresh
    if summary:
        summary_prompt = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_prompt = "Create a summary of the conversation above:"

    # Append the instruction as a HumanMessage after the current message list
    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    response = await llm.ainvoke(messages)

    # Keep only the 2 most recent messages; delete everything older
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    print(f"✅ Summary updated — kept 2 of {len(state['messages'])} messages")

    return {"summary": response.content, "messages": delete_messages}


# ---------------------------------------------------------------------------
# ROUTING
#
# Called after every call_model invocation to decide the next step.
# Replaces the built-in tools_condition so we can also route to "summarize".
#
# Priority order (important — tools must complete before we summarize):
#   1. Tool calls on last message → "tools"
#   2. Message count at threshold → "summarize"
#   3. Otherwise                  → END
# ---------------------------------------------------------------------------

def route_after_model(state: State):
    """
    Route after call_model:
      - Tool calls present           → "tools"
      - Message count at threshold   → "summarize"
      - Otherwise                    → END
    """
    messages = state["messages"]
    last = messages[-1]

    # Tool call takes priority — summarize only after all tools have completed
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # Enough messages accumulated — time to compact
    if len(messages) >= MESSAGE_SUMMARIZE_THRESHOLD:
        return "summarize"

    return END


# ---------------------------------------------------------------------------
# GRAPH SETUP (async, runs once at bot startup via post_init)
# ---------------------------------------------------------------------------

async def post_init(application):
    """
    Build and compile the LangGraph graph with SQLite checkpointing.
    Called by python-telegram-bot after the Application is initialised.
    """
    global graph
    conn = await aiosqlite.connect(
        os.path.join(os.path.dirname(__file__), "sessions.db")
    )
    checkpointer = AsyncSqliteSaver(conn)

    builder = StateGraph(State)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_node("summarize", summarize_conversation)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
    builder.add_edge("tools", "call_model")
    builder.add_edge("summarize", END)

    graph = builder.compile(checkpointer=checkpointer)
    print("✅ Agent graph with context compaction initialized")
    print(f"   Summarization threshold: {MESSAGE_SUMMARIZE_THRESHOLD} messages")


# ---------------------------------------------------------------------------
# TELEGRAM HANDLER
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context):
    """
    Main message handler. Two responsibilities:

    1. APPROVAL RESOLUTION: If the user has a pending command/deletion approval
       and replies YES or NO — resolve it immediately.

    2. NORMAL CHAT: All other messages run through the LangGraph agent.
       Summarization is handled automatically inside the graph when the
       message count threshold is reached — no special handling needed here.
    """
    user_message = update.message.text.strip()
    thread_id = str(update.effective_chat.id)

    # --- Check if this is a YES/NO approval response ---
    if thread_id in pending_approvals:
        command = pending_approvals[thread_id]

        if user_message.upper() == "YES":
            save_approval(command, approved=True)
            del pending_approvals[thread_id]
            print(f"✅ Approved: {command}")
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            output = result.stdout + result.stderr or "(no output)"
            await update.message.reply_text(f"✅ Ran: `{command}`\n\n{output}")
            return

        elif user_message.upper() == "NO":
            save_approval(command, approved=False)
            del pending_approvals[thread_id]
            print(f"❌ Denied: {command}")
            await update.message.reply_text(f"❌ Denied. `{command}` will not run.")
            return

    # --- Normal message — invoke the LangGraph agent ---
    set_current_thread_id(thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
    )
    await update.message.reply_text(response["messages"][-1].content)


# ---------------------------------------------------------------------------
# BOT STARTUP
# ---------------------------------------------------------------------------

app = (
    Application.builder()
    .token(os.getenv("TELEGRAM_BOT_TOKEN"))
    .post_init(post_init)
    .build()
)
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
