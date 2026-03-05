# OpenClaw Clone — Building a Personal AI Agent with LangGraph

A progressive, hands-on implementation of a personal AI agent — built step by step,
from a basic chatbot to a fully-featured assistant with persistent memory, tool use,
permission controls, multi-channel capability, and scheduled autonomous tasks.

This project is a clone of OpenClaw's architecture, rebuilt using:
- **LangGraph** — for stateful, graph-based agent workflows
- **SQLite** — for persistent conversation checkpoints
- **python-telegram-bot** — as the primary user interface
- **OpenAI GPT-4o** — as the underlying language model

## What We're Building

A personal AI agent (Jarvis) that:
- Remembers conversations across sessions (session memory)
- Compresses long histories automatically (context compaction)
- Retains key facts forever across session resets (long-term memory)
- Can read, write, and delete files on your Desktop
- Can run shell commands with permission controls
- Has multi-channel capability (Telegram + HTTP API)
- Can schedule recurring autonomous tasks (heartbeats)

## Project Structure

| File | Concept |
|---|---|
| `01_agent.py` | Bare minimum agent |
| `02_agentwithmemory.py` | In-memory conversation memory |
| `03_filememory.py` | SQLite-backed persistent memory |
| `04_memorywithsoul.py` | Personality via SOUL.md |
| `05_agentwithtool.py` | File and shell tools |
| `06_agentpermissioncontrol.py` | Safe/dangerous command classification |
| `07_gateway.py` | Multi-channel gateway (Telegram + HTTP) |
| `08_contextcompaction.py` | Automatic context window management |
| `09_longtermemory.py` | Cross-session persistent memory |
| `10_heartbeat.py` | Scheduled autonomous tasks |
| `agent_core.py` | Shared agent logic (used by 07) |
| `tools.py` | Shared tools and permission system (used by 07–10) |

---

## Prerequisites

- Python 3.10+
- A Telegram bot token (from [@BotFather](https://t.me/botfather))
- An OpenAI API key

## Installation

```bash
git clone <repo-url>
cd openclawclone
pip install -r requirements.txt

# For 10_heartbeat.py only (adds APScheduler for JobQueue)
pip install "python-telegram-bot[job-queue]"
```

## Environment Setup

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
```

## Project Files Overview

```
openclawclone/
├── 01_agent.py                  # Bare minimum agent
├── 02_agentwithmemory.py        # In-memory conversation memory
├── 03_filememory.py             # SQLite persistent memory
├── 04_memorywithsoul.py         # Personality via SOUL.md
├── 05_agentwithtool.py          # File and shell tools
├── 06_agentpermissioncontrol.py # Permission-controlled commands
├── 07_gateway.py                # Multi-channel gateway
├── 08_contextcompaction.py      # Context window management
├── 09_longtermemory.py          # Cross-session persistent memory
├── 10_heartbeat.py              # Scheduled autonomous tasks
├── agent_core.py                # Shared agent logic (used by 07)
├── tools.py                     # Shared tools (used by 07–10)
├── SOUL.md                      # Agent personality definition
├── TOOLS.md                     # Tool descriptions for agent context
├── schedules.json               # Persisted heartbeat schedules
├── sessions.db                  # SQLite conversation checkpoints
├── memory/                      # Long-term memory files
│   └── *.md
└── read_sessions.py             # Utility to inspect sessions.db
```

---

## Step 1 — `01_agent.py`: The Bare Minimum

The starting point. No memory, no tools, no personality — just a Telegram bot
that sends each message to GPT-4o and returns the reply.

```python
client = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))

async def handle_message(update: Update, context):
    user_message = update.message.text
    response = await client.ainvoke([HumanMessage(content=user_message)])
    await update.message.reply_text(response.content)
```

**What it does:**
- Each message is sent to the LLM as a fresh, standalone request
- No history — the agent has no memory of previous messages
- No state — every message is independent

**What it lacks:**
- The agent cannot remember anything you said earlier in the conversation
- Closing and reopening Telegram starts from zero every time

**Run it:**
```bash
python 01_agent.py
```

> This is the foundation. Every subsequent file builds on this by adding one
> new capability at a time.

---

## Step 2 — `02_agentwithmemory.py`: Conversation Memory

Introduces **LangGraph** and a checkpointer so the agent remembers the full
conversation history within a session.

**What changes from 01:**

Instead of a direct LLM call, messages now flow through a LangGraph `StateGraph`.
The graph stores every message in a checkpoint keyed by `thread_id` (the user's
Telegram chat ID).

```python
checkpointer = InMemorySaver()

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)
```

Each message handler passes the `thread_id` in a config dict — LangGraph uses
this to load and save the right checkpoint:

```python
thread_id = str(update.effective_chat.id)
config = {"configurable": {"thread_id": thread_id}}
response = await graph.ainvoke(
    {"messages": [{"role": "user", "content": user_message}]},
    config,
)
```

**What it does:**
- The agent remembers everything said earlier in the conversation
- Each user gets their own isolated history via `thread_id`
- Uses `InMemorySaver` — fast, but history is lost on bot restart

**What it still lacks:**
- Memory resets when the bot restarts (`InMemorySaver` is RAM-only)
- No personality or system prompt

> `InMemorySaver` → `AsyncSqliteSaver` is the only change needed to make
> memory survive restarts. That happens in the next file.

---

## Step 3 — `03_filememory.py`: Persistent SQLite Memory

The one critical upgrade from `02`: swap `InMemorySaver` for `AsyncSqliteSaver`.
Memory now survives bot restarts.

**The key change — async graph setup via `post_init`:**

SQLite requires an async connection, so the graph can no longer be built at import
time. Instead it's built inside `post_init` — a hook python-telegram-bot calls
after the Application starts:

```python
async def post_init(application):
    global graph
    conn = await aiosqlite.connect("sessions.db")
    checkpointer = AsyncSqliteSaver(conn)
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    graph = builder.compile(checkpointer=checkpointer)
```

```python
app = Application.builder()
    .token(os.getenv("TELEGRAM_BOT_TOKEN"))
    .post_init(post_init)   # wires the async setup
    .build()
```

**What changes from 02:**

| 02 | 03 |
|---|---|
| `InMemorySaver()` | `AsyncSqliteSaver(conn)` |
| Graph built at import time | Graph built in `post_init` |
| Memory lost on restart | Memory persists in `sessions.db` |

**What it does:**
- Full conversation history stored in `sessions.db`
- Bot can restart and pick up the conversation exactly where it left off
- Each user's history isolated by `thread_id` (their Telegram chat ID)

> `post_init` + `AsyncSqliteSaver` is the pattern used in every subsequent
> file. All later files extend this foundation.

---

## Step 4 — `04_memorywithsoul.py`: Personality via SOUL.md

Adds a system prompt so the agent has a consistent personality and name (Jarvis).
The personality is defined in a separate markdown file — `SOUL.md` — so it can be
edited without touching any Python code.

**Loading the personality:**

```python
with open(os.path.join(os.path.dirname(__file__), "SOUL.md"), "r") as f:
    SOUL = f.read()
```

**Injecting it as a system prompt on every call:**

```python
def call_model(state: MessagesState):
    messages = [{"role": "system", "content": SOUL}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

The system prompt is prepended fresh on every `call_model` invocation — it is
never stored in the checkpoint. This means you can edit `SOUL.md` and the change
takes effect on the next message without restarting the bot.

**`SOUL.md` structure:**

```markdown
# Who You Are
**Name:** Jarvis
**Role:** Personal AI assistant

## Personality
- Be genuinely helpful, not performatively helpful
- Skip the "Great question!" - just help
- Have opinions. You're allowed to disagree

## Boundaries
- Private things stay private
- When in doubt, ask before acting externally

## Memory
You have a long-term memory system that persists across sessions...
```

**What changes from 03:**

| 03 | 04 |
|---|---|
| No system prompt | `SOUL.md` injected on every call |
| Generic LLM responses | Named agent with consistent personality |
| Personality in code | Personality in editable markdown file |

> Every subsequent file uses this same pattern — loading personality from
> `SOUL.md` and injecting it in `call_model`. The SOUL is always the first
> message the LLM sees.

---

## Step 5 — `05_agentwithtool.py`: File and Shell Tools

Gives the agent the ability to act — read and write files on your Desktop.
This introduces LangGraph's tool loop: the model decides when to call a tool,
the `ToolNode` executes it, and the result flows back to the model.

**Defining tools with `@tool`:**

```python
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
```

**Registering tools with the LLM and graph:**

```python
tools = [read_desktop_file, write_desktop_file]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)  # LLM now knows the tool schemas
```

**The tool loop — new graph edges:**

```python
builder.add_node("tools", tool_node)
builder.add_conditional_edges("call_model", tools_condition)
# tools_condition: if last message has tool_calls → "tools", else → END
builder.add_edge("tools", "call_model")  # result flows back to model
```

```
START → call_model → [tool calls?] → tools → call_model → END
                   → [no tool calls] → END
```

**Also adds `TOOLS.md`** — a markdown file describing available tools, appended
to the system prompt so the agent knows what it can do:

```python
SYSTEM_PROMPT = SOUL + "\n\n" + TOOLS_DOC
```

**What changes from 04:**

| 04 | 05 |
|---|---|
| LLM only responds in text | LLM can call tools to take actions |
| `llm.invoke()` | `llm_with_tools.invoke()` |
| Single node graph | Two-node graph with tool loop |
| Only `SOUL.md` | `SOUL.md` + `TOOLS.md` as system prompt |

> The `@tool` decorator, `ToolNode`, and `tools_condition` pattern is the
> foundation for all tool use in files 06–10.

---

## Step 6 — `06_agentpermissioncontrol.py`: Permission Controls

Adds a safety layer for shell commands. Not all commands should run immediately —
`rm`, `sudo`, `curl` and others require explicit user approval via Telegram before
executing.

**Three-tier command classification:**

```python
SAFE_COMMANDS = {"ls", "cat", "head", "tail", "wc", "date", "whoami", "echo", "pwd"}

DANGEROUS_PATTERNS = [
    r"\brm\b",      # delete files
    r"\bsudo\b",    # superuser execution
    r"\bchmod\b",   # change permissions
    r"\bcurl\b",    # network requests
    r"\|.*sh\b",    # piping into shell
    ...
]

def check_command_safety(command: str) -> str:
    if base_cmd in SAFE_COMMANDS:        return "safe"             # run immediately
    if command in approvals["allowed"]:  return "approved"         # user said YES before
    if matches DANGEROUS_PATTERNS:       return "needs_approval"   # ask user
    return "needs_approval"  # unknown = ask anyway
```

**The approval flow:**

When a dangerous command is detected, the tool stores it in `pending_approvals`
and asks the user via Telegram. The next message from that user is checked first:

```python
pending_approvals: dict[str, str] = {}  # {thread_id: command}

if thread_id in pending_approvals:
    command = pending_approvals[thread_id]
    if user_message.upper() == "YES":
        save_approval(command, approved=True)   # persisted to exec-approvals.json
        del pending_approvals[thread_id]
        result = subprocess.run(command, ...)   # now runs
    elif user_message.upper() == "NO":
        save_approval(command, approved=False)
        del pending_approvals[thread_id]
```

**Approvals persist to `exec-approvals.json`** — once you approve a command,
you're never asked again.

**Also adds `delete_desktop_file`** as a dedicated tool — file deletion always
requires approval regardless of `SAFE_COMMANDS`:

```python
@tool
def delete_desktop_file(filename: str) -> str:
    """Delete a file from the Desktop. Always requires user approval."""
    pending_approvals[thread_id] = f"rm ~/Desktop/{filename}"
    return "⚠️ Reply YES to confirm or NO to cancel."
```

**What changes from 05:**

| 05 | 06 |
|---|---|
| Any tool runs immediately | Safe / approved / needs-approval classification |
| No user confirmation | YES/NO flow via Telegram before dangerous commands |
| No audit trail | `exec-approvals.json` persists decisions |
| No delete tool | `delete_desktop_file` always requires approval |

> `pending_approvals`, `save_approval`, and `check_command_safety` are extracted
> into `tools.py` in later files so all channels share the same permission system.

---

## Step 7 — `07_gateway.py`: Multi-Channel Gateway

The first major architectural shift. Instead of one file doing everything,
the code is split into three modules:

```
tools.py       — all tools + permission system (channel-agnostic)
agent_core.py  — LangGraph graph + run_turn() (channel-agnostic)
07_gateway.py  — Telegram + HTTP channels (thin layer, imports the above)
```

**`run_turn()` is the single entry point for all channels:**

```python
# agent_core.py
async def run_turn(thread_id: str, user_message: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
    )
    return response["messages"][-1].content
```

Any channel — Telegram, HTTP, Discord — calls `run_turn()` with a `thread_id`
and a message. The agent doesn't know or care which channel it came from.

**Shared memory across channels:**

Both channels use the same `sessions.db`. If a user's Telegram chat ID is passed
as `user_id` in HTTP calls, the conversation history is shared:

```bash
# This HTTP call continues the Telegram conversation
curl -X POST http://localhost:8080/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id": "8541638841", "message": "what did I ask earlier?"}'
```

**FastAPI runs in a separate thread** to avoid event loop conflicts with
python-telegram-bot:

```python
threading.Thread(target=run_api, daemon=True).start()

# Ctrl+C cleanly shuts down uvicorn:
def handle_sigint(sig, frame):
    server.should_exit = True
    original_sigint(sig, frame)
signal.signal(signal.SIGINT, handle_sigint)
```

**HTTP endpoints:**

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Send a message, get a response |
| `POST /approve` | Resolve a pending YES/NO command approval |
| `GET /pending/{user_id}` | Check if a command is awaiting approval |

> **Note:** macOS reserves port 5000 for AirPlay. Use port 8080 or higher.

**What changes from 06:**

| 06 | 07 |
|---|---|
| Single file, everything inline | Modular: tools.py + agent_core.py + gateway |
| Telegram only | Telegram + HTTP API |
| Permission logic in bot file | Shared `tools.py` used by all channels |

---

## Step 8 — `08_contextcompaction.py`: Context Window Management

**The problem:** After weeks of chatting, `sessions.db` accumulates thousands of
messages. LangGraph loads the full history on every call — eventually exceeding
GPT-4o's 128k token limit.

**The solution:** A native LangGraph `summarize` node that automatically compresses
old history and stores the summary in the checkpoint alongside messages.

**Extended state — `summary` field added:**

```python
class State(MessagesState):
    summary: str    # persisted in sessions.db, survives bot restarts
```

**The `summarize_conversation` node — incremental summarization:**

```python
async def summarize_conversation(state: State):
    summary = state.get("summary", "")

    # Extends prior summary rather than rewriting — context never lost
    if summary:
        summary_prompt = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_prompt = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    response = await llm.ainvoke(messages)   # base llm — no tools during summarization

    # Keep only the 2 most recent messages; delete everything older
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
```

**`call_model` injects the summary into the system prompt:**

```python
async def call_model(state: State):
    summary = state.get("summary", "")
    system_content = SYSTEM_PROMPT
    if summary:
        system_content += f"\n\n[Earlier in this session — summarized]\n{summary}"
    ...
```

**Custom routing replaces `tools_condition`:**

```python
MESSAGE_SUMMARIZE_THRESHOLD = 10   # lower to 4 for testing

def route_after_model(state: State):
    if hasattr(last, "tool_calls") and last.tool_calls:  return "tools"
    if len(messages) >= MESSAGE_SUMMARIZE_THRESHOLD:     return "summarize"
    return END
```

**Updated graph flow:**

```
START → call_model → route_after_model:
                       "tools"     → tools → call_model
                       "summarize" → summarize_conversation → END
                       END         → END
```

**What changes from 07:**

| Before | After |
|---|---|
| History grows forever | Summarized after every 10 messages |
| Risk of exceeding 128k window | Always well within context limit |
| `MessagesState` | `State(MessagesState)` with `summary: str` |
| `tools_condition` | Custom `route_after_model` |

> To test compaction quickly, set `MESSAGE_SUMMARIZE_THRESHOLD = 4`,
> have a short conversation, and observe:
> `✅ Summary updated — kept 2 of 10 messages`

---

## Step 9 — `09_longtermemory.py`: Cross-Session Persistent Memory

**The problem with session memory alone:** `sessions.db` is scoped to a
`thread_id`. Start a new session (new `thread_id`) and everything is gone —
the agent doesn't know your name, preferences, or past decisions.

**The solution:** A second memory layer — plain markdown files in `./memory/` —
that the agent explicitly saves to and that survive session resets forever.

**Two memory layers working together:**

```
sessions.db       → session memory   (this conversation, compressed by compaction)
./memory/*.md     → long-term memory (facts across all sessions, forever)
```

**Two new tools:**

```python
@tool
def save_memory(key: str, content: str) -> str:
    """Save important information to long-term memory."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(os.path.join(MEMORY_DIR, f"{key}.md"), "w") as f:
        f.write(content)
    return f"✅ Saved to long-term memory: '{key}'"

@tool
def memory_search(query: str) -> str:
    """Search long-term memory for relevant information."""
    # keyword search across all .md files
```

**Memory auto-loaded into every call — no agent initiative required:**

```python
def load_all_memories() -> str:
    """Read all ./memory/*.md files into a single string."""
    ...

async def call_model(state: State):
    system_content = SYSTEM_PROMPT

    long_term = load_all_memories()
    if long_term:
        system_content += f"\n\n[Long-term memory]\n{long_term}"

    summary = state.get("summary", "")
    if summary:
        system_content += f"\n\n[Earlier in this session — summarized]\n{summary}"
```

**Three context layers on every call:**

```
[1] SOUL + TOOLS instructions        — always
[2] ./memory/*.md (long-term facts)  — always, loaded fresh each call
[3] state["summary"] (session recap) — once compaction has run
```

**To verify it's working:**
1. Tell the agent your name → it saves to `./memory/user-name.md`
2. Delete `sessions.db` (wipe session history)
3. Ask "what's my name?" → agent still knows from `./memory/`

**What changes from 08:**

| 08 | 09 |
|---|---|
| Memory resets with new session | Key facts survive forever in `./memory/` |
| 4 tools | 6 tools (+ `save_memory`, `memory_search`) |
| Single memory layer | Two layers: session + long-term |

---

## Step 10 — `10_heartbeat.py`: Scheduled Autonomous Tasks

So far the agent only responds when you message it. Heartbeats let it act
autonomously on a timer — daily briefings, reminders, end-of-day summaries —
without any user prompt.

**The key insight:** scheduling is just another tool. The user asks in natural
language; the agent calls `create_schedule()`:

```
"remind me every day at 8am with a motivational quote"
      ↓
Agent calls: create_schedule("morning-quote", "08:00", "daily",
                              "Give me a motivational quote for the day")
      ↓
Saved to schedules.json → registered in JobQueue → fires at 08:00 daily
→ result sent back to the user's Telegram chat
```

**Three new scheduling tools:**

```python
@tool
def create_schedule(name: str, time: str, frequency: str, prompt: str) -> str:
    """Create a recurring scheduled task."""
    chat_id = get_current_thread_id()    # who to send the result to
    schedule = {"name": name, "time": time, "prompt": prompt, "chat_id": chat_id}
    save_schedules_file(schedules)        # persist to schedules.json
    register_job(_app, schedule)          # register in JobQueue immediately
    return f"✅ Scheduled '{name}' — will run daily at {time}."

@tool
def list_schedules() -> str:
    """List all active scheduled tasks."""

@tool
def delete_schedule(name: str) -> str:
    """Cancel and remove a scheduled task by name."""
```

**Schedules persist across restarts** via `schedules.json`. On startup,
`post_init` reloads every saved schedule back into the JobQueue:

```python
async def post_init(application):
    global _app
    _app = application           # gives scheduling tools access to JobQueue
    ...
    saved = load_schedules_file()
    for schedule in saved:
        register_job(application, schedule)   # re-register on every restart
```

**Each heartbeat runs in an isolated `cron:` thread_id** — never pollutes
the user's main conversation history:

```python
def make_heartbeat(name: str, prompt: str, chat_id: str):
    async def heartbeat(context):
        response = await run_turn_local(f"cron:{name}", prompt)  # isolated thread
        await context.bot.send_message(chat_id=chat_id, text=response)
    return heartbeat
```

**Uses python-telegram-bot's built-in `JobQueue`** (APScheduler) — runs
inside the existing asyncio event loop, no extra threads needed:

```python
app.job_queue.run_daily(heartbeat_fn, time=datetime.time(hour, minute), name=name)
```

**What changes from 09:**

| 09 | 10 |
|---|---|
| Agent only reacts to messages | Agent can act autonomously on a timer |
| Schedules hardcoded in code | User manages schedules via natural language |
| No persistence of schedules | `schedules.json` survives restarts |
| 6 tools | 9 tools (+ `create_schedule`, `list_schedules`, `delete_schedule`) |

> **Extra install required:**
> ```bash
> pip install "python-telegram-bot[job-queue]"
> ```

---

## Architecture Summary

```
User (Telegram / HTTP)
        ↓
  handle_message()
        ↓
  run_turn(thread_id, message)
        ↓
  graph.ainvoke()
     ↓         ↓
call_model   tools
     ↓         ↓
 summarize ←──┘
     ↓
sessions.db (checkpoint)
     +
./memory/*.md (long-term)
     +
schedules.json (heartbeats)
```

---

## Troubleshooting

**Bot doesn't respond**
- Check `.env` has `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY`
- Only one bot process can poll at a time — kill other running instances

**`sessions.db` errors on startup**
- Delete `sessions.db` and let LangGraph recreate it fresh

**Port already in use (07_gateway.py)**
- macOS reserves port 5000 for AirPlay — use port 8080 or higher
- `lsof -i :8080` to find what's using it

**Agent not saving memories (09, 10)**
- Check `./memory/` folder exists after first save
- Remind the agent explicitly: "save this to your memory"

**Heartbeats not firing (10)**
- Ensure `python-telegram-bot[job-queue]` is installed
- Check `schedules.json` exists and has correct `time` format (HH:MM)
- Verify the bot has been running past the scheduled time

**Compaction not triggering (08, 09, 10)**
- Lower `MESSAGE_SUMMARIZE_THRESHOLD = 4` temporarily to test
- Check console for `=== Summarizing conversation ===`
