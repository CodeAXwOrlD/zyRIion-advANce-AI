import subprocess
import asyncio
import edge_tts

VOICE = "en-IN-NeerjaExpressiveNeural"
TEMP_FILE = "/tmp/zyrion_tts.mp3"

async def generate_audio(text):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(TEMP_FILE)

def speak(text):
    print(f"🔊 ZYRION says: {text}")
    asyncio.run(generate_audio(text))
    subprocess.run(["mpg123", "-q", TEMP_FILE])

if __name__ == "__main__":
    speak("Hello Akhil! I am ZYRION, your personal AI assistant. I am online and ready to help you!")
