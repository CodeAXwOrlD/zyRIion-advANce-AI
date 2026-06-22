from collections import deque

MAX_TURNS = 6  # keeps last 6 messages = last 3 user+reply pairs

class ShortTermMemory:
    def __init__(self, max_turns=MAX_TURNS):
        self.history = deque(maxlen=max_turns)

    def add(self, user_text, assistant_text):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})

    def get_messages(self):
        return list(self.history)

    def clear(self):
        self.history.clear()
