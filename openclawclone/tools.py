"""
tools.py — All agent tools and the permission system.

This module is channel-agnostic. It knows nothing about Telegram or HTTP.
It exposes:
  - tools           : list of LangChain @tool functions
  - tool_node       : LangGraph ToolNode wrapping the tools list
  - pending_approvals: dict of commands awaiting user YES/NO
  - set_current_thread_id(): called before each graph invocation
  - save_approval() / load_approvals(): persist decisions to disk
"""

import os
import re
import json
import subprocess

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


# ---------------------------------------------------------------------------
# PERMISSION SYSTEM CONFIGURATION
#
# SAFE_COMMANDS: read-only shell commands that run immediately, no approval.
# DANGEROUS_PATTERNS: regex list — any match triggers the approval flow.
# APPROVALS_FILE: persists YES/NO decisions across restarts.
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
    r"\|.*sh\b",     # piping into shell
]

APPROVALS_FILE = os.path.join(os.path.dirname(__file__), "exec-approvals.json")

# ---------------------------------------------------------------------------
# DESKTOP PATH — all file tools are sandboxed here
# ---------------------------------------------------------------------------
DESKTOP = os.path.expanduser("~/Desktop")

# ---------------------------------------------------------------------------
# PENDING APPROVALS
#
# When a tool needs user approval, it stores the command here keyed by
# thread_id. The gateway (Telegram YES/NO or HTTP /approve) resolves it.
#
# Structure: { "telegram_123456": "rm ~/Desktop/notes.txt" }
# ---------------------------------------------------------------------------
pending_approvals: dict[str, str] = {}

# ---------------------------------------------------------------------------
# CURRENT THREAD ID
#
# @tool functions have no access to Telegram/HTTP context. agent_core calls
# set_current_thread_id() before every graph.ainvoke() so tools can read it.
# ---------------------------------------------------------------------------
_current_thread_id: str = ""


def set_current_thread_id(thread_id: str):
    """Set the active thread ID before invoking the graph."""
    global _current_thread_id
    _current_thread_id = thread_id


def get_current_thread_id() -> str:
    """Read the active thread ID from inside a tool."""
    return _current_thread_id


# ---------------------------------------------------------------------------
# APPROVAL HELPERS
# ---------------------------------------------------------------------------

def load_approvals() -> dict:
    """Load the persisted approvals/denials from disk."""
    if os.path.exists(APPROVALS_FILE):
        with open(APPROVALS_FILE) as f:
            return json.load(f)
    return {"allowed": [], "denied": []}


def save_approval(command: str, approved: bool):
    """Persist a YES/NO decision so the user is never asked twice."""
    approvals = load_approvals()
    key = "allowed" if approved else "denied"
    if command not in approvals[key]:
        approvals[key].append(command)
    with open(APPROVALS_FILE, "w") as f:
        json.dump(approvals, f, indent=2)


def check_command_safety(command: str) -> str:
    """
    Classify a shell command:
      'safe'           → SAFE_COMMANDS list, run immediately
      'approved'       → user previously approved, run immediately
      'needs_approval' → dangerous pattern or unknown, ask user
    """
    base_cmd = command.strip().split()[0] if command.strip() else ""

    if base_cmd in SAFE_COMMANDS:
        return "safe"

    approvals = load_approvals()
    if command in approvals["allowed"]:
        return "approved"

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return "needs_approval"

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
    Always requires explicit user approval — deletion is irreversible.
    If not yet approved, asks the user to reply YES or NO.
    """
    thread_id = get_current_thread_id()
    command = f"rm ~/Desktop/{filename}"

    # Already approved — delete immediately
    approvals = load_approvals()
    if command in approvals["allowed"]:
        path = os.path.join(DESKTOP, filename)
        if not os.path.exists(path):
            return f"File '{filename}' not found on Desktop."
        os.remove(path)
        return f"✅ Deleted '{filename}' from Desktop."

    # Needs approval — store and ask
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
    Safe commands run immediately. All others require user approval first.
    """
    thread_id = get_current_thread_id()
    safety = check_command_safety(command)

    if safety in ("safe", "approved"):
        print(f"✅ Running [{safety}]: {command}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr

    print(f"⚠️  Needs approval: {command}")
    pending_approvals[thread_id] = command
    return (
        f"⚠️ This command needs your approval:\n`{command}`\n\n"
        f"Reply YES to allow or NO to deny."
    )


# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------
tools = [read_desktop_file, write_desktop_file, delete_desktop_file, run_command]
tool_node = ToolNode(tools)
