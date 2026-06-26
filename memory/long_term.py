"""
ZYRION Long-Term Memory — Qdrant + mem0
Persistent semantic memory across restarts/days.
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import GROQ_API_KEY
from mem0 import Memory

DEFAULT_USER = "akhil"

CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "zyrion_memory",
            "embedding_model_dims": 384,
        },
    },
    "llm": {
        "provider": "groq",
        "config": {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "api_key": GROQ_API_KEY,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
}

_memory = None

def _get_memory():
    global _memory
    if _memory is None:
        print("[MEMORY] initialising (connecting to Qdrant)...")
        _memory = Memory.from_config(CONFIG)
        print("[MEMORY] ready")
    return _memory

def save(text: str, user_id: str = DEFAULT_USER):
    try:
        _get_memory().add(text, user_id=user_id)
        print(f"[MEMORY] saved: {text[:80]}")
    except Exception as e:
        print(f"[MEMORY] save failed: {e}")

def recall(query: str, user_id: str = DEFAULT_USER, limit: int = 3) -> list:
    try:
        results = _get_memory().search(query, filters={"user_id": user_id}, limit=limit)
        if isinstance(results, dict):
            results = results.get("results", [])
        memories = [r["memory"] for r in results if "memory" in r]
        if memories:
            print(f"[MEMORY] recalled {len(memories)} result(s) for: '{query[:50]}'")
        return memories
    except Exception as e:
        print(f"[MEMORY] recall failed: {e}")
        return []

def list_all(user_id: str = DEFAULT_USER) -> list:
    try:
        results = _get_memory().get_all(filters={"user_id": user_id})
        if isinstance(results, dict):
            results = results.get("results", [])
        return [r["memory"] for r in results if "memory" in r]
    except Exception as e:
        print(f"[MEMORY] list failed: {e}")
        return []

def clear(user_id: str = DEFAULT_USER):
    try:
        _get_memory().delete_all(user_id=user_id)
        print(f"[MEMORY] cleared all memories for {user_id}")
    except Exception as e:
        print(f"[MEMORY] clear failed: {e}")

if __name__ == "__main__":
    print("--- ZYRION long-term memory test ---\n")
    save("Akhil prefers short and direct replies")
    save("Akhil is building ZYRION on Linux Mint")
    save("Akhil likes dark mode and minimal UI")

    print("\nRecalling memories for 'display preferences':")
    for m in recall("display preferences"):
        print(f"  - {m}")

    print("\nAll stored memories:")
    for m in list_all():
        print(f"  - {m}")
