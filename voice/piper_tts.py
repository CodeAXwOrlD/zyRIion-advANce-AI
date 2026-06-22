import subprocess
import asyncio
import edge_tts
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import VOICE

# Persistent event loop — avoids asyncio.run() overhead on every speak() call
_loop = None

def _get_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop

async def _speak_streaming(text):
    """Stream TTS audio directly into mpg123 stdin — first audio byte plays
    within ~200ms instead of waiting for the full file to be generated."""
    process = subprocess.Popen(
        ["mpg123", "--quiet", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio" and process.stdin:
                process.stdin.write(chunk["data"])
    except Exception as e:
        print(f"❌ TTS streaming error: {e}")
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()

def speak(text):
    print(f"🔊 ZYRION says: {text}")
    t0 = time.time()
    loop = _get_loop()
    loop.run_until_complete(_speak_streaming(text))
    print(f"⏱️ TTS total (stream+play): {time.time() - t0:.2f}s")

if __name__ == "__main__":
    speak("Hello Akhil! I am ZYRION, your personal AI assistant. I am online and ready to help you!")
