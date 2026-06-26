import os
import sys
import sqlite3
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodic.db")

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            commands_count INTEGER DEFAULT 0,
            commands_summary TEXT
        )
    """)
    conn.commit()
    return conn

def start_session() -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO sessions (start_time) VALUES (?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
    )
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    print(f"[EPISODIC] session {session_id} started")
    return session_id

def end_session(session_id: int, commands: list):
    if not session_id:
        return
    summary = " | ".join(commands[:5])
    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET end_time=?, commands_count=?, commands_summary=? WHERE id=?",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(commands), summary, session_id)
    )
    conn.commit()
    conn.close()
    print(f"[EPISODIC] session {session_id} closed — {len(commands)} commands")

def get_recent_sessions(limit: int = 10) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, start_time, end_time, commands_count, commands_summary FROM sessions ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [{"id": r[0], "start": r[1], "end": r[2], "count": r[3], "summary": r[4]} for r in rows]

if __name__ == "__main__":
    sid = start_session()
    end_session(sid, ["open YouTube", "what's the weather", "play music"])
    for s in get_recent_sessions(5):
        print(f"Session {s['id']} | {s['start']} → {s['end']} | {s['count']} commands | {s['summary']}")
