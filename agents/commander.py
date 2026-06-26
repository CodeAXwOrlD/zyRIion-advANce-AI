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

SYSTEM_PROMPT = """You are the Commander Agent inside ZYRION, Akhil's personal AI assistant.
Read Akhil's spoken command and decide which specialized agent should handle it.
Available agents:
- researcher: web search, weather, news, wikipedia, stock/crypto prices
- tasks: Notion, Gmail, Calendar, Google Drive, Sheets
- laptop_control: open apps, browser, files, screenshot, volume, system control
- phone_control: calls, WhatsApp, phone apps, phone screenshot, hotspot
- social: Instagram, Snapchat, Telegram
- project_agent: writing/running/debugging code projects
- general: casual conversation or anything you can answer directly

Rules:
1. Pick the single best matching agent.
2. Reply MAXIMUM 8 WORDS. No filler.
   - general: answer directly in one short sentence.
   - others: acknowledge briefly, say not wired yet.

GOOD: "Opening YouTube.", "Done.", "It's 32 degrees in Jaipur."
BAD: "Alright, I will now proceed to open YouTube for you."

Respond ONLY with valid JSON:
{"agent": "<agent_name>", "reply": "<max 8 words>"}
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
            max_tokens=60
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
    test = "Hello Zyrion, how are you?"
    print(f"Testing: '{test}'")
    print(route_command(test))
