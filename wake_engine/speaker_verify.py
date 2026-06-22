import sounddevice as sd
import numpy as np
import os
import random
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from voice.whisper_stt import transcribe_audio
from wake_engine.speaker_embed import extract_embedding

SAMPLE_RATE = 16000
DURATION = 3
THRESHOLD = 0.82          # neural embeddings: legit ≈ 0.85-0.95, imposters ≈ 0.3-0.6
VOICE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "akhil_voice.npy")

CHALLENGE_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo",
    "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mango", "nova", "oscar",
    "papa", "romeo", "sierra", "tango", "zyrion"
]


def generate_challenge():
    """Pick 2 random words as the challenge phrase."""
    return random.sample(CHALLENGE_WORDS, 2)


def record_voice():
    try:
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return np.zeros(0, dtype='float32')


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def verify_speaker(challenge_words):
    """
    Returns (True/False, score).
    Checks BOTH neural voice embedding AND that the challenge words were spoken.
    """
    if not os.path.exists(VOICE_PATH):
        print("❌ No voice enrolled! Run enroll_voice.py first.")
        return False, 0.0

    saved = np.load(VOICE_PATH, allow_pickle=True).item()
    saved_embedding = saved['embedding']

    print("🎤 Speak now...")
    audio = record_voice()
    if len(audio) == 0:
        print("❌ No audio recorded.")
        return False, 0.0

    # Check 1: neural voice embedding match
    current_embedding = extract_embedding(audio)
    similarity = cosine_similarity(saved_embedding, current_embedding)
    print(f"🔍 Voice match score: {similarity:.4f} (threshold: {THRESHOLD})")

    if similarity < THRESHOLD:
        print(f"❌ Voice mismatch — not Akhil (score {similarity:.4f} < {THRESHOLD})")
        return False, similarity

    # Check 2: did they say the challenge words?
    print("📝 Checking challenge phrase...")
    spoken_text = transcribe_audio(audio)
    print(f"📢 Heard: '{spoken_text}'")

    spoken_lower = spoken_text.lower()
    words_heard = [w for w in challenge_words if w in spoken_lower]

    if len(words_heard) < 2:
        print(f"❌ Challenge failed — expected '{' '.join(challenge_words)}', heard '{spoken_text}'")
        return False, similarity

    print("✅ Voice + challenge both passed!")
    return True, similarity


if __name__ == "__main__":
    from voice.piper_tts import speak
    words = generate_challenge()
    phrase = " ".join(words)
    speak(f"Say these words: {phrase}")
    result, score = verify_speaker(words)
    print(f"Result: {result}, Score: {score:.4f}")
