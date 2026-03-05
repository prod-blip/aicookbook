"""
09_longtermemory.py — Telegram bot with session compaction + long-term memory.

Two memory layers:

  1. Session memory (sessions.db)
     - Conversation history for the current thread
     - Compacted automatically after MESSAGE_SUMMARIZE_THRESHOLD messages
     - Summary persists in SQLite across bot restarts
     - Scoped to a thread_id — a new session = blank slate

  2. Long-term memory (./memory/*.md)
     - Key facts the agent explicitly saves across ALL sessions
     - Survives session resets and bot restarts
     - The agent decides what's worth saving (user prefs, project context, etc.)
     - Loaded automatically into the system prompt on every call

How they work together:

  New session starts
        ↓
  load_all_memories() → injects ./memory/*.md into system prompt  ← long-term
        ↓
  User chats; messages accumulate in sessions.db                  ← session
        ↓
  10 messages → summarize_conversation fires                       ← compaction
        ↓
  Agent calls save_memory("key", "fact") for anything important   ← long-term
        ↓
  Next session starts fresh — but long-term memory is still there

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
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

# Import existing tools and permission infrastructure — memory tools added below
from tools import tools as base_tools, set_current_thread_id, pending_approvals, save_approval

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r") as f:
    SOUL = f.read()

with open(os.path.join(os.path.dirname(__file__), "TOOLS.md"), "r") as f:
    TOOLS_DOC = f.read()

# Appended to SOUL — tells the agent it has a long-term memory system
MEMORY_INSTRUCTIONS = """
## Long-Term Memory
You have a persistent memory system that survives session resets.

- Use **save_memory** to store important facts: user name, preferences, project details,
  decisions made, files created. Save proactively — if something seems worth remembering
  across conversations, save it.
- Use **memory_search** when you need to look for something specific that may have been
  saved in a previous session.
- Long-term memory is already loaded into your context at the start of every message —
  you do not need to call memory_search unless looking for something specific.
"""

SYSTEM_PROMPT = SOUL + "\n\n" + TOOLS_DOC + "\n\n" + MEMORY_INSTRUCTIONS

print("✅ SOUL, TOOLS, and MEMORY instructions loaded")


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY DIRECTORY
#
# Each memory is a markdown file: ./memory/{key}.md
# The key is a short label like "user-preferences" or "project-notes".
# Files are plain text — readable, editable, inspectable outside the bot.
# ---------------------------------------------------------------------------

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory")


def load_all_memories() -> str:
    """
    Read all memory files and return them as a single formatted string.
    Called on every call_model invocation so the agent always has full context.
    Returns an empty string if no memories exist yet.
    """
    if not os.path.exists(MEMORY_DIR):
        return ""

    memories = []
    for fname in sorted(os.listdir(MEMORY_DIR)):
        if fname.endswith(".md"):
            key = fname[:-3]  # strip .md
            with open(os.path.join(MEMORY_DIR, fname), "r") as f:
                content = f.read().strip()
            if content:
                memories.append(f"[{key}]\n{content}")

    return "\n\n".join(memories)


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY TOOLS
#
# save_memory: Agent writes important facts to disk.
# memory_search: Agent searches memory files by keyword (for targeted lookups).
#
# These are added to base_tools to form all_tools.
# ---------------------------------------------------------------------------

@tool
def save_memory(key: str, content: str) -> str:
    """
    Save important information to long-term memory.
    Use for user preferences, key facts, project details — anything worth
    remembering across sessions.

    key: short label, e.g. 'user-preferences', 'project-notes', 'user-name'
    content: the information to store (plain text or markdown)
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    filepath = os.path.join(MEMORY_DIR, f"{key}.md")
    with open(filepath, "w") as f:
        f.write(content)
    print(f"💾 Memory saved: {key}")
    return f"✅ Saved to long-term memory: '{key}'"


@tool
def memory_search(query: str) -> str:
    """
    Search long-term memory for relevant information.
    Use when looking for something specific from a previous session.
    Note: all memories are already loaded into context — use this only for
    targeted lookups when you need to confirm or find a specific fact.

    query: what to search for (keywords)
    """
    if not os.path.exists(MEMORY_DIR):
        return "No long-term memory exists yet."

    query_words = query.lower().split()
    results = []

    for fname in sorted(os.listdir(MEMORY_DIR)):
        if fname.endswith(".md"):
            with open(os.path.join(MEMORY_DIR, fname), "r") as f:
                content = f.read()
            # Match if any query word appears in the file
            if any(word in content.lower() for word in query_words):
                key = fname[:-3]
                results.append(f"[{key}]\n{content.strip()}")

    if results:
        return "\n\n".join(results)
    return "No matching memories found."


# ---------------------------------------------------------------------------
# LLM
#
# all_tools = base_tools (file ops + shell) + memory tools
# A new ToolNode and llm binding are created here so the graph uses all tools.
# ---------------------------------------------------------------------------

all_tools = base_tools + [save_memory, memory_search]
all_tool_node = ToolNode(all_tools)

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(all_tools)

graph = None


# ---------------------------------------------------------------------------
# STATE
#
# Extends MessagesState with "summary" for session compaction.
# Both fields are persisted in sessions.db via AsyncSqliteSaver.
# ---------------------------------------------------------------------------

class State(MessagesState):
    summary: str


# ---------------------------------------------------------------------------
# SUMMARIZATION THRESHOLD
#
# Lower to 4 for testing. Raise to 20–30 in production.
# ---------------------------------------------------------------------------

MESSAGE_SUMMARIZE_THRESHOLD = 10


# ---------------------------------------------------------------------------
# LANGGRAPH NODES
# ---------------------------------------------------------------------------

async def call_model(state: State):
    """
    Core LLM node.

    Injects three layers of context into the system prompt:
      1. SOUL + TOOLS personality and tool instructions (always)
      2. Long-term memory from ./memory/*.md (always — loaded fresh each call)
      3. Session summary from state["summary"] (once compaction has run)

    Reads:   state["messages"], state["summary"]
    Updates: state["messages"]
    """
    system_content = SYSTEM_PROMPT

    # Layer 2: inject long-term memory (cross-session facts)
    long_term = load_all_memories()
    if long_term:
        system_content += f"\n\n[Long-term memory — facts from previous sessions]\n{long_term}"

    # Layer 3: inject session summary (compressed current-session history)
    summary = state.get("summary", "")
    if summary:
        system_content += f"\n\n[Earlier in this session — summarized]\n{summary}"

    messages = [{"role": "system", "content": system_content}] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def summarize_conversation(state: State):
    """
    Summarization node. Triggered when message count reaches MESSAGE_SUMMARIZE_THRESHOLD.

    Incremental — each new summary extends the previous one.
    Deletes all but the 2 most recent messages after summarizing.

    Reads:   state["messages"], state["summary"]
    Updates: state["summary"], state["messages"] (via RemoveMessage)
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
    global graph
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
    print("✅ Agent graph with long-term memory initialized")
    print(f"   Memory dir:   {MEMORY_DIR}")
    print(f"   Compaction threshold: {MESSAGE_SUMMARIZE_THRESHOLD} messages")

    # Show how many memory files already exist
    if os.path.exists(MEMORY_DIR):
        files = [f for f in os.listdir(MEMORY_DIR) if f.endswith(".md")]
        print(f"   Loaded {len(files)} memory file(s): {[f[:-3] for f in files]}")
    else:
        print("   No memory files yet — agent will create them as needed")


# ---------------------------------------------------------------------------
# TELEGRAM HANDLER
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context):
    """
    Main message handler.

    1. YES/NO approval resolution for pending shell commands.
    2. Normal chat — routed through the LangGraph agent.
       Long-term memory and session compaction are transparent to this handler.
    """
    user_message = update.message.text.strip()
    thread_id = str(update.effective_chat.id)

    # --- YES/NO approval for pending commands ---
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
# ---------------------------------------------------------------------------

app = (
    Application.builder()
    .token(os.getenv("TELEGRAM_BOT_TOKEN"))
    .post_init(post_init)
    .build()
)
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
