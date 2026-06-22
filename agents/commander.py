import os
import sys
import json
from groq import Groq
import time 

# Add project root to sys.path dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import GROQ_API_KEY, LLM_MODEL

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
2. Write a short, natural spoken reply (1-2 sentences, conversational, not robotic).
   - If agent is "general": actually answer the question/chat normally.
   - If agent is anything else: acknowledge the command, briefly say that part isn't wired up yet.

Respond ONLY with valid JSON, nothing else:
{"agent": "<agent_name>", "reply": "<what to say out loud>"}
"""

def route_command(command_text, memory=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if memory:
        messages.extend(memory.get_messages())

    messages.append({"role": "user", "content": command_text})

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=200
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
