import os
import sys
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from groq import Groq
from core.config import GROQ_API_KEY, ROUTER_MODEL
from core.logger import log
from memory.long_term import save, recall

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are the Commander Agent inside ZYRION, Akhil's personal AI assistant running on Linux Mint / GNOME desktop.

Read Akhil's spoken command and decide which agent handles it, then return JSON.

Available agents:
- researcher: web search, weather, news, wikipedia, stock/crypto prices
- tasks: Notion, Gmail, Calendar, Google Drive, Sheets
- laptop_control: ANY laptop/desktop action — open apps, websites, folders, files, volume, brightness, screenshot, lock, sleep, shutdown, restart, run terminal commands, search the web
- phone_control: calls, WhatsApp, phone apps, phone screenshot, hotspot
- social: Instagram, Snapchat, Telegram
- project_agent: writing/running/debugging code projects
- general: casual conversation or anything you can answer directly

For laptop_control, also return "action" and "target" fields.
Available actions:
- open_app: open any desktop application. target = app binary name (e.g. "firefox", "code", "nautilus", "spotify", "vlc", "gedit", "gnome-terminal", "nemo")
- open_url: open a website. target = full URL (e.g. "https://youtube.com")
- open_folder: open a folder. target = path (e.g. "/home/indmadmax/Downloads")
- open_file: open a file. target = file path
- screenshot: take a screenshot. target = ""
- set_volume: control volume. target = "mute", "unmute", "up", "down", or a number like "50"
- set_brightness: control brightness. target = "up", "down", or a number like "70"
- lock_screen: lock the screen. target = ""
- sleep: suspend the system. target = ""
- shutdown: shut down. target = ""
- restart: restart. target = ""
- run_command: run any shell command. target = the shell command
- search_web: google search. target = search query

Rules:
- reply = max 8 words, natural spoken response, no filler
- For laptop_control: reply confirms the action briefly
- For general: answer directly
- For other agents not yet built: acknowledge briefly

GOOD replies: "Opening YouTube.", "Done.", "Volume set to 50%.", "Screenshot saved."
BAD replies: "Alright, I will now proceed to open YouTube for you."

Respond ONLY with valid JSON. Examples:

{"agent": "laptop_control", "action": "open_url", "target": "https://youtube.com", "reply": "Opening YouTube."}
{"agent": "laptop_control", "action": "open_app", "target": "code", "reply": "Opening VS Code."}
{"agent": "laptop_control", "action": "set_volume", "target": "50", "reply": "Volume set to 50%."}
{"agent": "laptop_control", "action": "screenshot", "target": "", "reply": "Screenshot taken."}
{"agent": "laptop_control", "action": "search_web", "target": "best Python tutorials", "reply": "Searching for Python tutorials."}
{"agent": "general", "reply": "I'm doing great, thanks!"}
{"agent": "researcher", "reply": "Researcher not wired yet."}
"""

def route_command(command_text: str, memory=None) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject long-term memories
    try:
        memories = recall(command_text)
        if memories:
            ctx = "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories)
            messages.append({"role": "system", "content": ctx})
            log.debug(f"[COMMANDER] {len(memories)} memories injected")
    except Exception as e:
        log.warning(f"[COMMANDER] memory recall skipped: {e}")

    # Inject short-term session memory
    if memory:
        messages.extend(memory.get_messages())

    messages.append({"role": "user", "content": command_text})

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=80
        )
        log.info(f"[COMMANDER] Groq LLM: {time.time() - t0:.2f}s")
        raw = response.choices[0].message.content.strip()

    except Exception as e:
        err = str(e).lower()
        log.error(f"[COMMANDER] LLM error: {e}")
        if "connection" in err or "network" in err or "timeout" in err:
            return {"agent": "general", "reply": "I'm offline right now, try again."}
        if "rate" in err or "limit" in err:
            return {"agent": "general", "reply": "Rate limited, try in a moment."}
        return {"agent": "general", "reply": "Something went wrong, try again."}

    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.startswith("```") and l.strip() != "json"]
        raw = "\n".join(lines).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"agent": "general", "reply": raw}

    # Save interaction to long-term memory
    try:
        save(f"Akhil said: {command_text}. ZYRION replied: {result['reply']}")
    except Exception as e:
        log.warning(f"[COMMANDER] memory save skipped: {e}")

    return result

if __name__ == "__main__":
    tests = [
        "open YouTube",
        "take a screenshot",
        "set volume to 70",
        "open my downloads folder",
        "search for latest AI news",
        "mute the volume",
    ]
    for t in tests:
        print(f"\n> {t}")
        print(route_command(t))
