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

TEMP_FILE = "/tmp/zyrion_tts.mp3"

async def generate_audio(text):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(TEMP_FILE)

def speak(text):
    print(f"🔊 ZYRION says: {text}")
    t0 = time.time()
    asyncio.run(generate_audio(text))
    print(f"⏱️ TTS generation: {time.time() - t0:.2f}s")
    try:
        t1 = time.time()
        subprocess.run(["mpg123", "-q", TEMP_FILE])
        print(f"⏱️ TTS playback: {time.time() - t1:.2f}s (this is just speech length, not lag)")
    finally:
        try:
            if os.path.exists(TEMP_FILE):
                os.remove(TEMP_FILE)
        except OSError:
            pass

if __name__ == "__main__":
    speak("Hello Akhil! I am ZYRION, your personal AI assistant. I am online and ready to help you!")
