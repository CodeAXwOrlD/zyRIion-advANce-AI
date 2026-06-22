import sounddevice as sd
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wake_engine.speaker_embed import extract_embedding

SAMPLE_RATE = 16000
DURATION = 5
NUM_SAMPLES = 5
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "akhil_voice.npy")


def record_sample(i):
    print(f"\n🎤 Recording sample {i+1}/{NUM_SAMPLES} — speak naturally for 5 seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    print("✅ Done!")
    return audio.flatten()


def enroll():
    print("🔐 ZYRION Voice Enrollment (Neural Embeddings)")
    print(f"Speak naturally for {DURATION} seconds each time\n")
    all_embeddings = []

    for i in range(NUM_SAMPLES):
        input("Press ENTER when ready...")
        audio = record_sample(i)
        embedding = extract_embedding(audio)
        all_embeddings.append(embedding)
        print(f"✅ Sample {i+1} embedded! (dim={len(embedding)})")

    # Average all embeddings into a single voiceprint
    avg_embedding = np.mean(all_embeddings, axis=0)

    # Normalise to unit vector for cosine similarity later
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    np.save(SAVE_PATH, {'embedding': avg_embedding})
    print(f"\n🔥 Neural voice fingerprint saved to {SAVE_PATH}")
    print("✅ Enrollment complete!")


if __name__ == "__main__":
    enroll()