import os
from dotenv import load_dotenv

# Find the project root directory dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VOICE = os.environ.get("VOICE", "en-IN-NeerjaExpressiveNeural")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "llama-3.1-8b-instant")