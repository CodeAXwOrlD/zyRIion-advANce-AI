import os
import sys
import json
from groq import Groq
import time 

# Add project root to sys.path dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import GROQ_API_KEY, ROUTER_MODEL

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are the Commander Agent inside ZYRION, Akhil's personal AI assistant.
Read Akhil's spoken command and decide which specialized agent should handle it.

Available agents:
- researcher: web search, weather, news, wikipedia, stock/crypto prices
- tasks: Notion, Gmail, Calendar, Google Drive, Sheets
- laptop_control: open apps, browser, files, screenshot, volume, system control
- phone_control: calls, WhatsApp, phone apps, phone screenshot, hotspot
- social: Instagram, Snapchat, Telegram
- project_agent: writing/running/debugging code projects
- general: casual conversation or anything you can just answer directly

Most agents besides "general" are not built yet. So:
1. Pick the single best matching agent from the list.
2. Write "reply" — MAXIMUM 8 WORDS. No filler, no explanation.
   - If agent is "general": answer in one very short sentence.
   - If agent is anything else: acknowledge briefly, say it's not wired up yet.

GOOD replies: "Opening YouTube.", "Done.", "Not wired up yet.", "It's 32 degrees in Jaipur."
BAD replies: "Alright, powering off. Please give it a few moments, it'll take about 8 seconds."

Respond ONLY with valid JSON, nothing else:
{"agent": "<agent_name>", "reply": "<max 8 words>"}
"""

def route_command(command_text, memory=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if memory:
        messages.extend(memory.get_messages())

    messages.append({"role": "user", "content": command_text})

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=60
        )
        print(f"⏱️ Groq LLM: {time.time() - t0:.2f}s")
        raw = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return {"agent": "general", "reply": "I'm sorry, I encountered an error communicating with my brain."}

    # Strip markdown code block fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"agent": "general", "reply": raw}

    return result

if __name__ == "__main__":
    test_command = "Hello Zyrion, how are you?"
    print(f"Testing route_command with: '{test_command}'")
    print(route_command(test_command))
