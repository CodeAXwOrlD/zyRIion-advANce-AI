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
from agents.laptop_agent import execute as laptop_execute
from memory.short_term import ShortTermMemory
from memory.episodic import start_session, end_session
from core.logger import log

SESSION_TIMEOUT = 45
MAX_VERIFY_ATTEMPTS = 3


def dispatch(result: dict) -> str:
    """Execute the action and return the final spoken reply."""
    agent = result.get("agent", "general")
    action = result.get("action", "")
    target = result.get("target", "")
    reply = result.get("reply", "Done.")

    if agent == "laptop_control" and action:
        executed_reply = laptop_execute(action, target)
        # Use executed reply only if it's more specific than the LLM reply
        return executed_reply if executed_reply else reply

    # All other agents not yet built — just speak the reply
    return reply


def verify_with_retries():
    for attempt in range(1, MAX_VERIFY_ATTEMPTS + 1):
        challenge_words = generate_challenge()
        phrase = " ".join(challenge_words)
        speak(f"Identify yourself. Say: {phrase}" if attempt == 1 else f"Try again. Say: {phrase}")
        passed, score = verify_speaker(challenge_words)
        if passed:
            record_success()
            return True
        record_failure(score=score, attempt=attempt)
        log.warning(f"[AUTH] attempt {attempt}/{MAX_VERIFY_ATTEMPTS} failed (score={score:.4f})")
    return False


def on_wake():
    if is_locked_out():
        remaining = lockout_remaining()
        log.warning(f"[SECURITY] locked out — {remaining // 60}m {remaining % 60}s remaining")
        return

    is_akhil = verify_with_retries()
    if not is_akhil:
        log.warning(f"[AUTH] all {MAX_VERIFY_ATTEMPTS} attempts failed")
        speak("Access denied.")
        return

    speak("Welcome Akhil.")
    memory = ShortTermMemory()
    session_id = start_session()
    session_commands = []
    session_start = time.time()
    first_command = True

    while time.time() - session_start < SESSION_TIMEOUT:
        command = listen_and_transcribe(skip_calibration=not first_command)
        first_command = False

        if not command or len(command.strip()) < 2:
            log.debug("[LISTEN] no speech detected — still listening...")
            continue

        log.info(f"[COMMAND] {command}")
        result = route_command(command, memory=memory)
        log.info(f"[ROUTE] agent={result['agent']} action={result.get('action','')} target={result.get('target','')}")

        spoken_reply = dispatch(result)
        speak(spoken_reply)

        memory.add(command, spoken_reply)
        session_commands.append(command)
        session_start = time.time()

    end_session(session_id, session_commands)
    speak("Going to sleep.")
    log.info("[SESSION] closed — back to clap-only listening\n")


def start_zyrion():
    log.info("ZYRION online — listening for activation")
    log.info("Double clap to activate\n")
    try:
        listen_for_claps(on_wake)
    except Exception as e:
        log.error(f"[ERROR] clap detection loop: {e}")


if __name__ == "__main__":
    start_zyrion()
