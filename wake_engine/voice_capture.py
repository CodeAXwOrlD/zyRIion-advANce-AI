import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wake_engine.clap_detection import listen_for_claps
from wake_engine.speaker_verify import verify_speaker, generate_challenge
from wake_engine.security_monitor import record_failure, record_success, is_locked_out, lockout_remaining
from voice.whisper_stt import listen_and_transcribe
from voice.piper_tts import speak
from agents.commander import route_command
from memory.short_term import ShortTermMemory

SESSION_TIMEOUT = 45
MAX_VERIFY_ATTEMPTS = 3


def verify_with_retries():
    for attempt in range(1, MAX_VERIFY_ATTEMPTS + 1):
        challenge_words = generate_challenge()
        phrase = " ".join(challenge_words)

        if attempt == 1:
            speak(f"Identify yourself. Say: {phrase}")
        else:
            speak(f"Try again. Say: {phrase}")

        passed, score = verify_speaker(challenge_words)

        if passed:
            record_success()
            return True

        record_failure(score=score, attempt=attempt)
        print(f"[AUTH] attempt {attempt}/{MAX_VERIFY_ATTEMPTS} failed (score={score:.4f})")

    return False


def on_wake():
    # Enforce lockout before any verification
    if is_locked_out():
        remaining = lockout_remaining()
        print(f"[SECURITY] locked out — {remaining // 60}m {remaining % 60}s remaining, ignoring clap")
        return

    is_akhil = verify_with_retries()

    if not is_akhil:
        print(f"[AUTH] all {MAX_VERIFY_ATTEMPTS} attempts failed — staying locked")
        speak("Access denied.")
        return

    speak("Welcome Akhil.")
    memory = ShortTermMemory()
    session_start = time.time()
    first_command = True

    while time.time() - session_start < SESSION_TIMEOUT:
        command = listen_and_transcribe(skip_calibration=not first_command)
        first_command = False

        if not command or len(command.strip()) < 2:
            print("[LISTEN] no speech detected — still listening...")
            continue

        print(f"\n[COMMAND] received: {command}")
        result = route_command(command, memory=memory)
        print(f"[ROUTE] agent: {result['agent']}")
        speak(result["reply"])

        memory.add(command, result["reply"])
        session_start = time.time()

    speak("Going to sleep.")
    print("[SESSION] closed — back to clap-only listening\n")


def start_zyrion():
    print("ZYRION online — listening for activation")
    print("Double clap to activate\n")
    try:
        listen_for_claps(on_wake)
    except Exception as e:
        print(f"[ERROR] clap detection loop: {e}")
        print("Check microphone connection and permissions.")


if __name__ == "__main__":
    start_zyrion()
