import sounddevice as sd
import numpy as np
import whisper

SAMPLE_RATE = 16000
DURATION = 6

model = None

def load_model():
    global model
    if model is None:
        print("⏳ Loading Whisper model...")
        model = whisper.load_model("base")
        print("✅ Whisper ready!")

def record_command():
    print("🎤 Listening for command...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording done!")
    return audio.flatten()

def transcribe(audio):
    load_model()
    audio_float = audio.astype(np.float32)
    result = model.transcribe(audio_float, language="en")
    text = result["text"].strip()
    print(f"📝 You said: {text}")
    return text

def listen_and_transcribe():
    audio = record_command()
    return transcribe(audio)

if __name__ == "__main__":
    load_model()
    text = listen_and_transcribe()
    print(f"Result: {text}")
EOF
