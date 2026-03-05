"""
07_gateway.py — Multi-channel gateway.

Two channels share the same agent, memory, and permission system:

  1. Telegram  — classic bot interface, YES/NO approval via reply
  2. HTTP API  — FastAPI on port 8080, approval via POST /approve

Both channels call run_turn() from agent_core.py. The agent doesn't know
which channel it's talking to — it just sees messages and thread IDs.

Thread ID convention (shared memory across channels):
  All channels for the same user share one thread_id.
  Telegram uses the chat_id mapped to a user identity.
  HTTP callers pass their user_id directly.
  This means a conversation started on Telegram continues seamlessly over HTTP.

Usage:
  python 07_gateway.py

  # Chat via HTTP:
  curl -X POST http://localhost:8080/chat \
       -H "Content-Type: application/json" \
       -d '{"user_id": "atul", "message": "list my desktop files"}'

  # Approve a pending command via HTTP:
  curl -X POST http://localhost:8080/approve \
       -H "Content-Type: application/json" \
       -d '{"user_id": "atul", "decision": "YES"}'
"""

import os
import asyncio
import subprocess
import threading
import signal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

from agent_core import setup_graph, run_turn
from tools import pending_approvals, save_approval

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
api = FastAPI(title="Agent Gateway")


class ChatRequest(BaseModel):
    user_id: str    # caller-supplied identifier, e.g. "atul"
    message: str


class ApproveRequest(BaseModel):
    user_id: str
    decision: str   # "YES" or "NO"


@api.post("/chat")
async def http_chat(req: ChatRequest):
    """
    Send a message to the agent from any HTTP client.
    If the agent's response triggers a permission request, the response
    will include 'pending_approval' with the command awaiting approval.
    """
    # Use user_id directly — shared with Telegram session for the same user
    thread_id = req.user_id
    response_text = await run_turn(thread_id, req.message)

    # Expose any pending approval so the client knows to call /approve
    pending = pending_approvals.get(thread_id)
    return {
        "response": response_text,
        "pending_approval": pending,   # None if no approval needed
    }


@api.post("/approve")
async def http_approve(req: ApproveRequest):
    """
    Resolve a pending approval for an HTTP user.
    Call this after /chat returns a non-null 'pending_approval'.
    """
    thread_id = req.user_id

    if thread_id not in pending_approvals:
        raise HTTPException(status_code=400, detail="No pending approval for this user.")

    command = pending_approvals[thread_id]

    if req.decision.upper() == "YES":
        save_approval(command, approved=True)
        del pending_approvals[thread_id]
        print(f"✅ HTTP approved: {command}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return {"result": result.stdout + result.stderr or "(no output)"}

    else:
        save_approval(command, approved=False)
        del pending_approvals[thread_id]
        print(f"❌ HTTP denied: {command}")
        return {"result": f"Denied. '{command}' will not run."}


@api.get("/pending/{user_id}")
async def http_pending(user_id: str):
    """Check if an HTTP user has a command awaiting approval."""
    return {"pending_approval": pending_approvals.get(user_id)}


# ---------------------------------------------------------------------------
# TELEGRAM HANDLER
# ---------------------------------------------------------------------------

async def handle_telegram_message(update: Update, context):
    """
    Handle an incoming Telegram message.

    Two cases:
      1. Pending approval → user replied YES/NO → resolve it
      2. Normal message   → pass to agent via run_turn()
    """
    user_message = update.message.text.strip()
    # Use the Telegram chat ID as the thread_id directly.
    # If you want this to share memory with HTTP, pass the same ID as user_id in HTTP calls.
    thread_id = str(update.effective_chat.id)

    # --- Approval resolution ---
    if thread_id in pending_approvals:
        command = pending_approvals[thread_id]

        if user_message.upper() == "YES":
            save_approval(command, approved=True)
            del pending_approvals[thread_id]
            print(f"✅ Telegram approved: {command}")
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            output = result.stdout + result.stderr or "(no output)"
            await update.message.reply_text(f"✅ Done.\n\n{output}")
            return

        elif user_message.upper() == "NO":
            save_approval(command, approved=False)
            del pending_approvals[thread_id]
            print(f"❌ Telegram denied: {command}")
            await update.message.reply_text(f"❌ Cancelled.")
            return

    # --- Normal agent turn ---
    response_text = await run_turn(thread_id, user_message)
    await update.message.reply_text(response_text)


# ---------------------------------------------------------------------------
# STARTUP
#
# post_init runs once after the Telegram app is ready but before polling
# starts. We use it to:
#   1. Initialize the LangGraph agent + SQLite connection
#   2. Start the FastAPI server as an asyncio task (same event loop)
# ---------------------------------------------------------------------------

async def post_init(application):
    """Initialize agent graph and start the HTTP API."""
    await setup_graph()

    # Run uvicorn in a separate thread. We keep a reference to the server
    # so we can signal it to exit when Ctrl+C is pressed.
    config = uvicorn.Config(api, host="0.0.0.0", port=8080, log_level="warning")
    server = uvicorn.Server(config)

    def run_api():
        server.run()

    threading.Thread(target=run_api, daemon=True).start()

    # On Ctrl+C, tell uvicorn to stop before the process exits
    original_sigint = signal.getsignal(signal.SIGINT)
    def handle_sigint(sig, frame):
        server.should_exit = True
        original_sigint(sig, frame)
    signal.signal(signal.SIGINT, handle_sigint)

    print("✅ HTTP API running on http://localhost:8080")
    print("✅ Telegram bot polling started")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

telegram_app = (
    Application.builder()
    .token(os.getenv("TELEGRAM_BOT_TOKEN"))
    .post_init(post_init)
    .build()
)
telegram_app.add_handler(MessageHandler(filters.TEXT, handle_telegram_message))
telegram_app.run_polling()
