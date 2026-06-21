import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
from groq import Groq
import sys
sys.path.append('/media/indmadmax/drive/zyrion')
from core.config import GROQ_API_KEY, WHISPER_MODEL

SAMPLE_RATE = 16000
DURATION = 8

client = Groq(api_key=GROQ_API_KEY)

def record_command():
    print("🎤 Listening for command...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording done!")
    return audio.flatten()

def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        with open(f.name, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language="en"
            )
    text = result.text.strip()
    print(f"📝 You said: {text}")
    return text

def listen_and_transcribe():
    audio = record_command()
    return transcribe(audio)

if __name__ == "__main__":
    text = listen_and_transcribe()
    print(f"Result: {text}")