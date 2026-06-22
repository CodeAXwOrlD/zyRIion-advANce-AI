import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
from groq import Groq
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import GROQ_API_KEY, WHISPER_MODEL

SAMPLE_RATE = 16000
DURATION = 8

client = Groq(api_key=GROQ_API_KEY)

# Module-level cached threshold from last calibration (reused within a session)
_cached_threshold = None

def transcribe_audio(audio):
    """Transcribe already-recorded numpy audio (used by speaker_verify challenge check)."""
    t1 = time.time()
    text = transcribe(audio)
    print(f"⏱️ Whisper API: {time.time() - t1:.2f}s")
    return text

def record_command(skip_calibration=False):
    global _cached_threshold
    print("🎤 Listening for command...")
    chunk_duration = 0.1
    chunk_samples = int(chunk_duration * SAMPLE_RATE)
    silence_needed = 3           # 0.3s of silence = done (was 0.6s)
    max_chunks = int(8 / chunk_duration)

    recorded = []
    silence_count = 0
    speech_started = False

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        stream.start()

        if skip_calibration and _cached_threshold is not None:
            # Reuse previous threshold — skip warmup+calibration entirely
            threshold = _cached_threshold
            print(f"🔧 Reusing cached threshold: {threshold:.4f}")
        else:
            # Warmup: 1 chunk to flush hardware buffer
            chunk, _ = stream.read(chunk_samples)
            recorded.append(chunk.flatten())

            # Calibrate: 2 chunks (0.2s)
            noise_samples = []
            for _ in range(2):
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk.flatten()
                recorded.append(chunk)
                noise_samples.append(np.sqrt(np.mean(chunk**2)))

            noise_floor = float(np.median(noise_samples))
            threshold = max(noise_floor * 3, 0.015)
            _cached_threshold = threshold
            print(f"🔧 Noise floor: {noise_floor:.4f} → speech threshold: {threshold:.4f}")

        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.flatten()
            recorded.append(chunk)

            volume = np.sqrt(np.mean(chunk**2))

            if volume > threshold:
                speech_started = True
                silence_count = 0
            elif speech_started:
                silence_count += 1
                if silence_count >= silence_needed:
                    break

        stream.stop()
        stream.close()
    except Exception as e:
        print(f"❌ Audio recording error: {e}")
        return np.zeros(0, dtype='float32')

    print("✅ Recording done!")
    return np.concatenate(recorded) if recorded else np.zeros(0, dtype='float32')

def transcribe(audio):
    if len(audio) == 0:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        try:
            with open(f.name, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio_file,
                    language="en"
                )
            text = result.text.strip()
            print(f"📝 You said: {text}")
            return text
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
        finally:
            try:
                os.remove(f.name)
            except OSError:
                pass

def listen_and_transcribe(skip_calibration=False):
    t0 = time.time()
    audio = record_command(skip_calibration=skip_calibration)
    print(f"⏱️ Recording: {time.time() - t0:.2f}s")

    t1 = time.time()
    text = transcribe(audio)
    print(f"⏱️ Whisper API: {time.time() - t1:.2f}s")
    return text

if __name__ == "__main__":
    text = listen_and_transcribe()
    print(f"Result: {text}")
