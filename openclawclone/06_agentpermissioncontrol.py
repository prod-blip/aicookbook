import os
import re
import json
import subprocess
import aiosqlite
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

print("✅ Token loaded:", bool(os.getenv("TELEGRAM_BOT_TOKEN")))

# ---------------------------------------------------------------------------
# SOUL + TOOLS: Load personality and tool descriptions from markdown files.
# These get injected as the system prompt on every LLM call.
# ---------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r") as f:
    SOUL = f.read()

with open(os.path.join(os.path.dirname(__file__), "TOOLS.md"), "r") as f:
    TOOLS_DOC = f.read()

SYSTEM_PROMPT = SOUL + "\n\n" + TOOLS_DOC

print("✅ SOUL and TOOLS loaded")

# ---------------------------------------------------------------------------
# PERMISSION SYSTEM CONFIGURATION
#
# SAFE_COMMANDS: These run immediately with no approval needed.
#   - Read-only, non-destructive shell commands.
#
# DANGEROUS_PATTERNS: Regex patterns that flag a command as high-risk.
#   - Any match → user must explicitly approve before execution.
#
# APPROVALS_FILE: JSON file that persists approved/denied commands across
#   restarts. Once you approve a command, you're never asked again.
# ---------------------------------------------------------------------------
SAFE_COMMANDS = {"ls", "cat", "head", "tail", "wc", "date", "whoami", "echo", "pwd"}

DANGEROUS_PATTERNS = [
    r"\brm\b",       # delete files
    r"\bsudo\b",     # superuser execution
    r"\bchmod\b",    # change file permissions
    r"\bchown\b",    # change file ownership
    r"\bmv\b",       # move/rename (can overwrite)
    r"\bkill\b",     # kill processes
    r"\bcurl\b",     # network requests
    r"\bwget\b",     # network downloads
    r"\|.*sh\b",     # piping into shell (dangerous)
]

APPROVALS_FILE = os.path.join(os.path.dirname(__file__), "exec-approvals.json")

# ---------------------------------------------------------------------------
# DESKTOP PATH: All file operations are sandboxed to the Desktop only.
# ---------------------------------------------------------------------------
DESKTOP = os.path.expanduser("~/Desktop")

# ---------------------------------------------------------------------------
# IN-MEMORY PENDING APPROVALS
#
# When a command needs approval, we store it here keyed by thread_id
# (Telegram chat ID). The next YES/NO message from that user resolves it.
#
# Structure: { "thread_id": "command string" }
# ---------------------------------------------------------------------------
pending_approvals: dict[str, str] = {}

# ---------------------------------------------------------------------------
# CURRENT THREAD ID
#
# The run_command tool needs to store the pending approval keyed by thread_id,
# but @tool functions don't receive Telegram context. We set this global just
# before every graph.ainvoke() call so the tool can read it.
# ---------------------------------------------------------------------------
current_thread_id: str = ""


# ---------------------------------------------------------------------------
# APPROVAL HELPERS
# ---------------------------------------------------------------------------

def load_approvals() -> dict:
    """Load the persistent approvals list from disk."""
    if os.path.exists(APPROVALS_FILE):
        with open(APPROVALS_FILE) as f:
            return json.load(f)
    return {"allowed": [], "denied": []}


def save_approval(command: str, approved: bool):
    """Persist a user's approval/denial so we never ask twice."""
    approvals = load_approvals()
    key = "allowed" if approved else "denied"
    if command not in approvals[key]:
        approvals[key].append(command)
    with open(APPROVALS_FILE, "w") as f:
        json.dump(approvals, f, indent=2)


def check_command_safety(command: str) -> str:
    """
    Classify a command into one of three categories:
      - 'safe'           → in SAFE_COMMANDS, run immediately
      - 'approved'       → user previously approved it, run immediately
      - 'needs_approval' → unknown or matches a dangerous pattern, ask user
    """
    base_cmd = command.strip().split()[0] if command.strip() else ""

    # Check if it's a known safe command
    if base_cmd in SAFE_COMMANDS:
        return "safe"

    # Check if the user has already approved this exact command before
    approvals = load_approvals()
    if command in approvals["allowed"]:
        return "approved"

    # Check for dangerous patterns (rm, sudo, curl | sh, etc.)
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return "needs_approval"

    # Unknown command — ask anyway (safe default)
    return "needs_approval"


# ---------------------------------------------------------------------------
# TOOLS
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


@tool
def delete_desktop_file(filename: str) -> str:
    """
    Delete a file from the user's Desktop.
    Always requires explicit user approval before deletion — this is irreversible.
    If approval is pending, instructs the user to reply YES or NO.
    """
    thread_id = current_thread_id
    command = f"rm ~/Desktop/{filename}"

    # Check if already approved
    approvals = load_approvals()
    if command in approvals["allowed"]:
        path = os.path.join(DESKTOP, filename)
        if not os.path.exists(path):
            return f"File '{filename}' not found on Desktop."
        os.remove(path)
        return f"✅ Deleted '{filename}' from Desktop."

    # Always ask for approval before deleting
    print(f"⚠️  Deletion needs approval: {filename}")
    pending_approvals[thread_id] = command
    return (
        f"⚠️ You've asked me to delete '{filename}'. This cannot be undone.\n\n"
        f"Reply YES to confirm or NO to cancel."
    )


@tool
def run_command(command: str) -> str:
    """
    Run a shell command on the user's machine.
    Safe commands execute immediately. Dangerous commands require explicit
    user approval via Telegram before running.
    """
    # Read the current thread_id from the global set before graph.ainvoke()
    thread_id = current_thread_id
    safety = check_command_safety(command)

    if safety in ("safe", "approved"):
        # Run immediately — no approval needed
        print(f"✅ Running command [{safety}]: {command}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr

    # Needs approval — store in pending dict and ask the user
    print(f"⚠️  Command needs approval: {command}")
    pending_approvals[thread_id] = command
    return (
        f"⚠️ This command needs your approval before I can run it:\n"
        f"`{command}`\n\n"
        f"Please reply YES to allow or NO to deny."
    )


tools = [read_desktop_file, write_desktop_file, delete_desktop_file, run_command]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

graph = None


# ---------------------------------------------------------------------------
# LANGGRAPH NODES
# ---------------------------------------------------------------------------

def call_model(state: MessagesState):
    """
    Core LLM node. Prepends the system prompt (SOUL + TOOLS) to every call
    so personality and tool awareness are always in context.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# GRAPH SETUP (async, initialised in post_init)
#
# Flow:
#   START → call_model → [tool_calls?] → tools → call_model → END
#                      → [no tool_calls] → END
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
    builder.add_edge("tools", "call_model")
    graph = builder.compile(checkpointer=checkpointer)
    print("✅ Graph with permission-controlled tools initialized")


# ---------------------------------------------------------------------------
# TELEGRAM HANDLERS
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context):
    """
    Main message handler. Two responsibilities:

    1. APPROVAL RESOLUTION: If the user's chat has a pending command approval
       and the message is YES/NO — resolve it.
         - YES → save approval, run command, send result
         - NO  → save denial, inform user

    2. NORMAL CHAT: All other messages go through the LangGraph agent.
    """
    user_message = update.message.text.strip()
    thread_id = str(update.effective_chat.id)

    # --- Check if this is a YES/NO approval response ---
    if thread_id in pending_approvals:
        command = pending_approvals[thread_id]

        if user_message.upper() == "YES":
            # User approved — save it and run the command
            save_approval(command, approved=True)
            del pending_approvals[thread_id]
            print(f"✅ User approved command: {command}")

            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            output = result.stdout + result.stderr or "(no output)"
            await update.message.reply_text(f"✅ Ran: `{command}`\n\n{output}")
            return

        elif user_message.upper() == "NO":
            # User denied — save it and inform
            save_approval(command, approved=False)
            del pending_approvals[thread_id]
            print(f"❌ User denied command: {command}")
            await update.message.reply_text(f"❌ Denied. `{command}` will not run.")
            return

    # --- Normal message — invoke the LangGraph agent ---
    # Set the global thread_id so run_command can read it inside the tool
    global current_thread_id
    current_thread_id = thread_id

    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
    )

    await update.message.reply_text(response["messages"][-1].content)


# ---------------------------------------------------------------------------
# BOT STARTUP
# ---------------------------------------------------------------------------

app = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).post_init(post_init).build()
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
