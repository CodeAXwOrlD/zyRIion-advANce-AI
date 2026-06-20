import sounddevice as sd
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
import os

SAMPLE_RATE = 16000
DURATION = 5
SAVE_PATH = os.path.expanduser("~/Documents/zyrion/wake_engine/akhil_voice.npy")

def record_sample(i):
    print(f"\n🎤 Recording sample {i+1}/5 — speak for 5 seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    print("✅ Done!")
    return audio.flatten()

def enroll():
    print("🔐 ZYRION Voice Enrollment — Akhil only!")
    print("Speak naturally for 5 seconds each time\n")
    encoder = VoiceEncoder()
    embeddings = []

    for i in range(5):
        input("Press ENTER when ready...")
        audio = record_sample(i)
        wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
        embedding = encoder.embed_utterance(wav)
        embeddings.append(embedding)
        print(f"✅ Sample {i+1} saved!")

    final_embedding = np.mean(embeddings, axis=0)
    np.save(SAVE_PATH, final_embedding)
    print(f"\n🔥 Voice fingerprint saved to {SAVE_PATH}")
    print("✅ Enrollment complete! ZYRION now knows your voice.")

if __name__ == "__main__":
    enroll()

