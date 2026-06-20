import sounddevice as sd
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import os

SAMPLE_RATE = 16000
DURATION = 3
THRESHOLD = 0.68
VOICE_PATH = os.path.expanduser("~/Documents/zyrion/wake_engine/akhil_voice.npy")

def record_voice():
    print("🎤 Speak now...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def verify_speaker():
    if not os.path.exists(VOICE_PATH):
        print("❌ No voice enrolled! Run enroll_voice.py first.")
        return False

    encoder = VoiceEncoder()
    saved_embedding = np.load(VOICE_PATH)

    audio = record_voice()
    wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
    current_embedding = encoder.embed_utterance(wav)

    similarity = np.dot(saved_embedding, current_embedding) / (
        np.linalg.norm(saved_embedding) * np.linalg.norm(current_embedding)
    )

    print(f"🔍 Voice match score: {similarity:.2f}")

    if similarity >= THRESHOLD:
        print("✅ Welcome Akhil!")
        return True
    else:
        print("❌ Access Denied!")
        return False

if __name__ == "__main__":
    verify_speaker()
