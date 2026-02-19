import json
import threading
import os

class MemoryManager:
    CACHE_VERSION = 3

    def __init__(self, cache_path='cache.jsonl'):
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self._cache = {}
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("v") != self.CACHE_VERSION:
                        continue
                    key = entry.get('input')
                    if key:
                        self._cache[key] = entry.get('ir')
                except Exception:
                    continue

    def get_ir(self, user_input):
        return self._cache.get(user_input)

    def add_ir(self, user_input, ir):
        self._cache[user_input] = ir
        threading.Thread(target=self._append_to_file, args=(user_input, ir), daemon=True).start()

    def _append_to_file(self, user_input, ir):
        with self.lock:
            with open(self.cache_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'v': self.CACHE_VERSION, 'input': user_input, 'ir': ir}, ensure_ascii=False) + '\n')

    def clear_cache(self):
        with self.lock:
            self._cache = {}
            with open(self.cache_path, "w", encoding="utf-8") as f:
                f.write("")
