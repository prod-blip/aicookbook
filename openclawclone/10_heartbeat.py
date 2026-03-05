"""
10_heartbeat.py — Telegram bot with scheduled heartbeats (agent-managed).

Builds on 09_longtermemory.py (tools + compaction + long-term memory) and adds
scheduled tasks that the agent can create, list, and delete via natural language.

The user never needs to edit the Python file to manage schedules — they just chat:

  "remind me every day at 8am with a motivational quote"
        ↓
  Agent calls create_schedule("morning-quote", "08:00", "daily",
                               "Give me a motivational quote")
        ↓
  Saved to schedules.json → registered in JobQueue → fires at 8am daily
  → result sent back to the user's Telegram chat

Key design decisions:
  - Schedules are persisted in schedules.json (survive bot restarts)
  - Each heartbeat uses an isolated thread_id "cron:<name>" in sessions.db
    so it never pollutes the user's main conversation history
  - The chat_id of whoever created the schedule is stored, so the result
    goes back to the right person
  - python-telegram-bot's built-in JobQueue (APScheduler) handles timing
    inside the existing asyncio event loop — no background threads needed

Requirements:
  pip install "python-telegram-bot[job-queue]"   # adds APScheduler
"""

import os
import json
import datetime
import subprocess
import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

from tools import (
    tools as base_tools,
    set_current_thread_id,
    get_current_thread_id,
    pending_approvals,
    save_approval,
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r") as f:
    SOUL = f.read()

with open(os.path.join(os.path.dirname(__file__), "TOOLS.md"), "r") as f:
    TOOLS_DOC = f.read()

SCHEDULE_INSTRUCTIONS = """
## Scheduled Tasks (Heartbeats)
You can create recurring tasks that fire automatically on a timer.

- Use **create_schedule** to schedule a daily task.
  Provide: name (short label), time (HH:MM, 24h), frequency ("daily"), prompt (what to do).
- Use **list_schedules** to show all active schedules.
- Use **delete_schedule** to cancel a schedule by name.

When a heartbeat fires, you run the prompt autonomously and send the result to the user.
Examples: daily briefings, reminders, end-of-day summaries, habit nudges.
"""

MEMORY_INSTRUCTIONS = """
## Long-Term Memory
You have a persistent memory system that survives session resets.
- Use **save_memory** to store important facts proactively.
- Use **memory_search** when looking for something specific from a previous session.
- Long-term memory is already loaded into your context on every call.
"""

SYSTEM_PROMPT = SOUL + "\n\n" + TOOLS_DOC + "\n\n" + MEMORY_INSTRUCTIONS + "\n\n" + SCHEDULE_INSTRUCTIONS

print("✅ SOUL, TOOLS, MEMORY, and SCHEDULE instructions loaded")


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY
# ---------------------------------------------------------------------------

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory")


def load_all_memories() -> str:
    """Read all memory files and return as a formatted string."""
    if not os.path.exists(MEMORY_DIR):
        return ""
    memories = []
    for fname in sorted(os.listdir(MEMORY_DIR)):
        if fname.endswith(".md"):
            key = fname[:-3]
            with open(os.path.join(MEMORY_DIR, fname), "r") as f:
                content = f.read().strip()
            if content:
                memories.append(f"[{key}]\n{content}")
    return "\n\n".join(memories)


@tool
def save_memory(key: str, content: str) -> str:
    """
    Save important information to long-term memory.
    Use for user preferences, key facts, project details — anything worth
    remembering across sessions.
    key: short label e.g. 'user-preferences', 'project-notes'
    content: the information to store
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(os.path.join(MEMORY_DIR, f"{key}.md"), "w") as f:
        f.write(content)
    print(f"💾 Memory saved: {key}")
    return f"✅ Saved to long-term memory: '{key}'"


@tool
def memory_search(query: str) -> str:
    """
    Search long-term memory for relevant information.
    Use when looking for something specific from a previous session.
    """
    if not os.path.exists(MEMORY_DIR):
        return "No long-term memory exists yet."
    query_words = query.lower().split()
    results = []
    for fname in sorted(os.listdir(MEMORY_DIR)):
        if fname.endswith(".md"):
            with open(os.path.join(MEMORY_DIR, fname), "r") as f:
                content = f.read()
            if any(word in content.lower() for word in query_words):
                results.append(f"[{fname[:-3]}]\n{content.strip()}")
    return "\n\n".join(results) if results else "No matching memories found."


# ---------------------------------------------------------------------------
# SCHEDULES PERSISTENCE
#
# schedules.json stores all active schedules so they survive bot restarts.
# Format: list of {name, time, frequency, prompt, chat_id}
# ---------------------------------------------------------------------------

SCHEDULES_FILE = os.path.join(os.path.dirname(__file__), "schedules.json")


def load_schedules_file() -> list:
    """Load saved schedules from disk."""
    if os.path.exists(SCHEDULES_FILE):
        with open(SCHEDULES_FILE, "r") as f:
            return json.load(f)
    return []


def save_schedules_file(schedules: list):
    """Persist schedules to disk."""
    with open(SCHEDULES_FILE, "w") as f:
        json.dump(schedules, f, indent=2)


# ---------------------------------------------------------------------------
# GLOBAL APP REFERENCE
#
# create_schedule and delete_schedule need access to the JobQueue to register
# or remove jobs at runtime. _app is set in post_init after the Application
# is built — tools read it via this module-level reference.
# ---------------------------------------------------------------------------

_app: Application = None


# ---------------------------------------------------------------------------
# HEARTBEAT RUNNER
#
# Each scheduled job calls run_turn_local with an isolated cron thread_id.
# The result is sent to the chat_id that created the schedule.
# ---------------------------------------------------------------------------

async def run_turn_local(thread_id: str, user_message: str) -> str:
    """Invoke the agent for a scheduled heartbeat (no Telegram context needed)."""
    set_current_thread_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
    )
    return response["messages"][-1].content


def make_heartbeat(name: str, prompt: str, chat_id: str):
    """
    Return an async job function for the JobQueue.
    Each heartbeat:
      1. Runs the prompt through the agent (isolated cron thread_id)
      2. Sends the result back to the user's chat
    """
    async def heartbeat(context):
        print(f"\n⏰ Heartbeat firing: {name}")
        try:
            response = await run_turn_local(f"cron:{name}", prompt)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⏰ *{name}*\n\n{response}",
                parse_mode="Markdown",
            )
            print(f"✅ Heartbeat sent: {name} → {chat_id}")
        except Exception as e:
            print(f"❌ Heartbeat error ({name}): {e}")
    return heartbeat


def register_job(app: Application, schedule: dict):
    """Register a schedule dict as a JobQueue job."""
    hour, minute = map(int, schedule["time"].split(":"))
    app.job_queue.run_daily(
        make_heartbeat(schedule["name"], schedule["prompt"], schedule["chat_id"]),
        time=datetime.time(hour, minute),
        name=schedule["name"],
    )
    print(f"✅ Job registered: '{schedule['name']}' at {schedule['time']} daily")


# ---------------------------------------------------------------------------
# SCHEDULING TOOLS
# ---------------------------------------------------------------------------

@tool
def create_schedule(name: str, time: str, frequency: str, prompt: str) -> str:
    """
    Create a recurring scheduled task (heartbeat).
    The task runs automatically at the specified time and sends the result
    back to the user who created it.

    name: short label e.g. 'morning-briefing', 'evening-summary'
    time: time in HH:MM 24-hour format e.g. '07:30', '21:00'
    frequency: how often to run — currently only 'daily' is supported
    prompt: what the agent should do when the heartbeat fires
    """
    # Store who created this schedule so the result goes back to them
    chat_id = get_current_thread_id()

    schedule = {
        "name": name,
        "time": time,
        "frequency": frequency,
        "prompt": prompt,
        "chat_id": chat_id,
    }

    # Save to disk (persist across restarts)
    schedules = load_schedules_file()
    schedules = [s for s in schedules if s["name"] != name]  # replace if exists
    schedules.append(schedule)
    save_schedules_file(schedules)

    # Register in JobQueue immediately (no restart needed)
    if _app:
        # Remove any existing job with the same name first
        for job in _app.job_queue.get_jobs_by_name(name):
            job.schedule_removal()
        register_job(_app, schedule)

    print(f"📅 Schedule created: '{name}' at {time} daily")
    return f"✅ Scheduled '{name}' — will run daily at {time}."


@tool
def list_schedules() -> str:
    """List all active scheduled tasks."""
    schedules = load_schedules_file()
    if not schedules:
        return "No scheduled tasks. Use create_schedule to add one."
    lines = []
    for s in schedules:
        lines.append(f"• *{s['name']}* — {s['frequency']} at {s['time']}\n  Prompt: {s['prompt']}")
    return "\n\n".join(lines)


@tool
def delete_schedule(name: str) -> str:
    """
    Cancel and remove a scheduled task by name.
    name: the label used when creating the schedule
    """
    schedules = load_schedules_file()
    remaining = [s for s in schedules if s["name"] != name]

    if len(remaining) == len(schedules):
        return f"No schedule named '{name}' found."

    save_schedules_file(remaining)

    # Remove from JobQueue if running
    if _app:
        for job in _app.job_queue.get_jobs_by_name(name):
            job.schedule_removal()

    print(f"🗑️  Schedule deleted: '{name}'")
    return f"✅ Deleted schedule '{name}'."


# ---------------------------------------------------------------------------
# LLM + ALL TOOLS
# ---------------------------------------------------------------------------

all_tools = base_tools + [save_memory, memory_search, create_schedule, list_schedules, delete_schedule]
all_tool_node = ToolNode(all_tools)

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(all_tools)

graph = None


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

class State(MessagesState):
    summary: str


MESSAGE_SUMMARIZE_THRESHOLD = 10


# ---------------------------------------------------------------------------
# LANGGRAPH NODES
# ---------------------------------------------------------------------------

async def call_model(state: State):
    """
    Core LLM node. Injects three context layers into the system prompt:
      1. SOUL + TOOLS + MEMORY + SCHEDULE instructions (always)
      2. Long-term memory from ./memory/*.md (always)
      3. Session summary from state["summary"] (after compaction runs)

    Reads:   state["messages"], state["summary"]
    Updates: state["messages"]
    """
    system_content = SYSTEM_PROMPT

    long_term = load_all_memories()
    if long_term:
        system_content += f"\n\n[Long-term memory]\n{long_term}"

    summary = state.get("summary", "")
    if summary:
        system_content += f"\n\n[Earlier in this session — summarized]\n{summary}"

    messages = [{"role": "system", "content": system_content}] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def summarize_conversation(state: State):
    """
    Summarization node. Triggered at MESSAGE_SUMMARIZE_THRESHOLD.
    Incremental — extends the existing summary rather than rewriting it.
    Keeps only the 2 most recent messages verbatim.

    Reads:   state["messages"], state["summary"]
    Updates: state["summary"], state["messages"]
    """
    print("\n=== Summarizing conversation ===")

    summary = state.get("summary", "")
    if summary:
        summary_prompt = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_prompt = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    response = await llm.ainvoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    print(f"✅ Summary updated — kept 2 of {len(state['messages'])} messages")

    return {"summary": response.content, "messages": delete_messages}


# ---------------------------------------------------------------------------
# ROUTING
# ---------------------------------------------------------------------------

def route_after_model(state: State):
    """
    Route after call_model:
      - Tool calls present         → "tools"
      - Message count at threshold → "summarize"
      - Otherwise                  → END
    """
    messages = state["messages"]
    last = messages[-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    if len(messages) >= MESSAGE_SUMMARIZE_THRESHOLD:
        return "summarize"

    return END


# ---------------------------------------------------------------------------
# GRAPH SETUP
# ---------------------------------------------------------------------------

async def post_init(application):
    """
    Build the LangGraph graph and reload all saved schedules into JobQueue.
    Called once by python-telegram-bot after Application is initialized.
    """
    global graph, _app
    _app = application  # give scheduling tools access to JobQueue

    conn = await aiosqlite.connect(
        os.path.join(os.path.dirname(__file__), "sessions.db")
    )
    checkpointer = AsyncSqliteSaver(conn)

    builder = StateGraph(State)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", all_tool_node)
    builder.add_node("summarize", summarize_conversation)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
    builder.add_edge("tools", "call_model")
    builder.add_edge("summarize", END)

    graph = builder.compile(checkpointer=checkpointer)
    print("✅ Agent graph initialized")

    # Reload saved schedules so heartbeats survive bot restarts
    saved = load_schedules_file()
    if saved:
        for schedule in saved:
            register_job(application, schedule)
        print(f"✅ Reloaded {len(saved)} saved schedule(s)")
    else:
        print("   No saved schedules")


# ---------------------------------------------------------------------------
# TELEGRAM HANDLER
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context):
    """
    Main message handler.
    1. YES/NO approval resolution for pending shell commands.
    2. Normal chat — routed through the LangGraph agent.
    """
    user_message = update.message.text.strip()
    thread_id = str(update.effective_chat.id)

    # --- YES/NO approval ---
    if thread_id in pending_approvals:
        command = pending_approvals[thread_id]

        if user_message.upper() == "YES":
            save_approval(command, approved=True)
            del pending_approvals[thread_id]
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr or "(no output)"
            await update.message.reply_text(f"✅ Ran: `{command}`\n\n{output}")
            return

        elif user_message.upper() == "NO":
            save_approval(command, approved=False)
            del pending_approvals[thread_id]
            await update.message.reply_text(f"❌ Denied. `{command}` will not run.")
            return

    # --- Normal message ---
    set_current_thread_id(thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
    )
    await update.message.reply_text(response["messages"][-1].content)


# ---------------------------------------------------------------------------
# BOT STARTUP
#
# Requires: pip install "python-telegram-bot[job-queue]"
# ---------------------------------------------------------------------------

app = (
    Application.builder()
    .token(os.getenv("TELEGRAM_BOT_TOKEN"))
    .post_init(post_init)
    .build()
)
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
