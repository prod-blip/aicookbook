import sqlite3
import msgpack

conn = sqlite3.connect("sessions.db")
threads = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()

for (thread_id,) in threads:
    print(f"\n{'='*50}")
    print(f"Thread ID (Chat ID): {thread_id}")

    row = conn.execute(
        "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY rowid DESC LIMIT 1",
        (thread_id,)
    ).fetchone()

    if not row:
        continue

    data = msgpack.unpackb(row[0], raw=False)
    channel_values = data.get("channel_values", {})

    # Show the summary if compaction has run for this thread
    summary = channel_values.get("summary", "")
    if summary:
        print(f"\n  [SUMMARY]\n  {summary}\n")

    messages = channel_values.get("messages", [])

    for msg in messages:
        if isinstance(msg, msgpack.ext.ExtType):
            decoded = msgpack.unpackb(msg.data, raw=False)
            # decoded = [module, class_name, {content, type, ...}, method]
            msg_data = decoded[2]
            role = msg_data.get("type", "unknown")
            content = msg_data.get("content", "")
            print(f"  [{role}]: {content}")

conn.close()
