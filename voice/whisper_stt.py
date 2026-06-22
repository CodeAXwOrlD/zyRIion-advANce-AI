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

def record_command():
    print("🎤 Listening for command...")
    chunk_duration = 0.1
    chunk_samples = int(chunk_duration * SAMPLE_RATE)
    warmup_chunks = 2          # discard stream-startup transient — don't let it skew calibration
    calibration_chunks = 4
    silence_needed = int(0.6 / chunk_duration)
    max_chunks = int(8 / chunk_duration)

    recorded = []
    silence_count = 0
    speech_started = False

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        stream.start()

        # Warm-up: still recorded (in case you start talking instantly), just excluded from calibration
        for _ in range(warmup_chunks):
            chunk, _ = stream.read(chunk_samples)
            recorded.append(chunk.flatten())

        # Calibrate against real steady-state noise, not the startup glitch
        noise_samples = []
        for _ in range(calibration_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.flatten()
            recorded.append(chunk)
            noise_samples.append(np.sqrt(np.mean(chunk**2)))

        noise_floor = float(np.median(noise_samples))  # median resists one-off spikes
        threshold = max(noise_floor * 3, 0.015)
        print(f"🔧 Noise floor: {noise_floor:.4f} → speech threshold: {threshold:.4f}")

        remaining_chunks = max_chunks - warmup_chunks - calibration_chunks
        for _ in range(remaining_chunks):
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

def listen_and_transcribe():
    t0 = time.time()
    audio = record_command()
    print(f"⏱️ Recording: {time.time() - t0:.2f}s")

    t1 = time.time()
    text = transcribe(audio)
    print(f"⏱️ Whisper API: {time.time() - t1:.2f}s")
    return text

if __name__ == "__main__":
    text = listen_and_transcribe()
    print(f"Result: {text}")
