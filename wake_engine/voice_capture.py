import threading
import sys
import os

sys.path.append(os.path.expanduser("~/Documents/zyrion"))

from wake_engine.clap_detection import listen_for_claps
from wake_engine.speaker_verify import verify_speaker
from voice.whisper_stt import listen_and_transcribe
from voice.piper_tts import speak

def on_wake():
    speak("Identify yourself")
    is_akhil = verify_speaker()

    if is_akhil:
        speak("Welcome Akhil! What can I do for you?")
        command = listen_and_transcribe()
        print(f"\n🧠 Command received: {command}")
        speak(f"Got it! You said: {command}")
    else:
        print("🔇 Stranger detected — staying silent")

def start_zyrion():
    print("🌌 ZYRION is awake and watching...")
    print("👏 Clap twice to activate\n")
    listen_for_claps(on_wake)

if __name__ == "__main__":
    start_zyrion()
