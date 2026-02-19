import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from core.listener import submit_shadow_text


BRIDGE_EVENTS_PATH = Path("memory/vscode_bridge_events.jsonl")
FORWARD_ASSISTANT = os.getenv("COGNITIVE_FORWARD_ASSISTANT", "0").strip().lower() in {"1", "true", "yes", "on"}


class VSCodeChatBridgeService:
    def __init__(self, poll_interval: float = 1.5, lookback_days: int = 30):
        self.poll_interval = max(0.5, float(poll_interval))
        self.lookback_days = max(1, int(lookback_days))
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._offsets: Dict[str, int] = {}
        self._seen: set[str] = set()
        self._stats = {
            "forwarded_user": 0,
            "forwarded_assistant": 0,
            "last_error": "",
            "watched_files": 0,
            "watched_paths": [],
            "started_at": "",
        }

    def start(self):
        with self._lock:
            if self._running:
                return self
            self._running = True
            self._stats["started_at"] = datetime.now(timezone.utc).isoformat()
            self._thread = threading.Thread(target=self._run, daemon=True, name="vscode-chat-bridge")
            self._thread.start()
        return self

    def stop(self):
        with self._lock:
            self._running = False

    def is_running(self):
        return bool(self._running and self._thread and self._thread.is_alive())

    def status(self):
        return {
            "running": self.is_running(),
            **self._stats,
        }

    def _run(self):
        while self._running:
            try:
                files = self._discover_chat_session_files()
                self._stats["watched_files"] = len(files)
                self._stats["watched_paths"] = [str(p) for p in files[:5]]
                for f in files:
                    self._poll_file(f)
            except Exception as exc:
                self._stats["last_error"] = str(exc)
                self._event("bridge_error", f"[DASHBOARD_ALERT] VSCode bridge error: {exc}")
            time.sleep(self.poll_interval)

    def _discover_chat_session_files(self) -> List[Path]:
        roots = [
            Path(os.path.expandvars(r"%APPDATA%\Code\User\workspaceStorage")),
            Path(os.path.expandvars(r"%APPDATA%\Code - Insiders\User\workspaceStorage")),
            Path(os.path.expandvars(r"%APPDATA%\Cursor\User\workspaceStorage")),
            Path.home() / ".codex" / "sessions",
        ]

        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        files: List[Path] = []
        all_candidates: List[Path] = []
        for root in roots:
            if not root.exists():
                continue
            if str(root).endswith(str(Path(".codex") / "sessions")):
                for p in root.rglob("*.jsonl"):
                    all_candidates.append(p)
            else:
                # Fast/common layout
                for p in root.glob("*\\chatSessions\\*.jsonl"):
                    all_candidates.append(p)
                # Fallback recursive layout
                for p in root.rglob("chatSessions/*.jsonl"):
                    all_candidates.append(p)

        # De-duplicate
        deduped: List[Path] = []
        seen = set()
        for p in all_candidates:
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            deduped.append(p)

        # Primary filter: recent files only
        for p in deduped:
            try:
                if datetime.fromtimestamp(p.stat().st_mtime) >= cutoff:
                    files.append(p)
            except Exception:
                continue

        # Fallback: if none found, keep latest files regardless of age
        if not files:
            by_mtime = []
            for p in deduped:
                try:
                    by_mtime.append((p.stat().st_mtime, p))
                except Exception:
                    continue
            by_mtime.sort(reverse=True, key=lambda x: x[0])
            files = [p for _, p in by_mtime[:20]]

        return files

    def _poll_file(self, path: Path):
        p = str(path)
        try:
            size = path.stat().st_size
        except Exception:
            return

        if p not in self._offsets:
            # Start from end to avoid replaying old sessions.
            self._offsets[p] = size
            return

        offset = self._offsets[p]
        if offset > size:
            offset = 0

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(offset)
                data = f.read()
                self._offsets[p] = f.tell()
        except Exception:
            return

        if not data.strip():
            return

        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            self._handle_line(line, path)

    def _handle_line(self, line: str, path: Path):
        try:
            obj = json.loads(line)
        except Exception:
            return

        # Codex session format:
        # {"type":"response_item","payload":{"type":"message","role":"user|assistant","content":[...]}}
        # {"type":"event_msg","payload":{"type":"user_message|agent_message","message":"..."}}
        evt_type = obj.get("type")
        payload = obj.get("payload")
        if evt_type == "response_item" and isinstance(payload, dict):
            if payload.get("type") == "message":
                role = payload.get("role")
                content = payload.get("content")
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        text = item.get("text")
                        if not isinstance(text, str):
                            continue
                        if role == "assistant":
                            self._forward_unique(text, f"codex_assistant:{path.name}", role="assistant")
                        elif role == "user":
                            self._forward_unique(text, f"codex_user:{path.name}", role="user")
            return
        if evt_type == "event_msg" and isinstance(payload, dict):
            ptype = payload.get("type")
            message = payload.get("message")
            if isinstance(message, str):
                if ptype == "agent_message":
                    self._forward_unique(message, f"codex_assistant:{path.name}", role="assistant")
                elif ptype == "user_message":
                    self._forward_unique(message, f"codex_user:{path.name}", role="user")
            return

        kind = obj.get("kind")
        key_path = obj.get("k", [])
        val = obj.get("v")

        # Assistant responses: {"kind":1|2, "k":["requests", N, "response"], "v":[{"value":"..."}]}
        if kind in (1, 2) and isinstance(key_path, list) and len(key_path) >= 3 and key_path[0] == "requests" and key_path[-1] == "response":
            texts = []
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        t = item.get("value")
                        if isinstance(t, str):
                            texts.append(t)
            for t in texts:
                self._forward_unique(t, f"vscode_assistant:{path.name}", role="assistant")
            return

        # User/assistant in requests batch: {"kind":1|2, "k":["requests"], "v":[{...}]}
        if kind in (1, 2) and isinstance(key_path, list) and key_path == ["requests"] and isinstance(val, list):
            for req in val:
                if not isinstance(req, dict):
                    continue
                message = req.get("message", {})
                text = message.get("text") if isinstance(message, dict) else None
                if isinstance(text, str):
                    self._forward_unique(text, f"vscode_user:{path.name}", role="user")
                result = req.get("result")
                if isinstance(result, dict):
                    rmsg = result.get("message")
                    if isinstance(rmsg, dict):
                        rtext = rmsg.get("text")
                        if isinstance(rtext, str):
                            self._forward_unique(rtext, f"vscode_assistant:{path.name}", role="assistant")
                    response = req.get("response")
                    if isinstance(response, list):
                        for item in response:
                            if isinstance(item, dict):
                                t = item.get("value")
                                if isinstance(t, str):
                                    self._forward_unique(t, f"vscode_assistant:{path.name}", role="assistant")
            return

        # Assistant updates can also arrive as {"k":["requests",N,"result"],"v":{...}}
        if kind in (1, 2) and isinstance(key_path, list) and len(key_path) >= 3 and key_path[0] == "requests" and key_path[-1] == "result":
            if isinstance(val, dict):
                rmsg = val.get("message")
                if isinstance(rmsg, dict):
                    rtext = rmsg.get("text")
                    if isinstance(rtext, str):
                        self._forward_unique(rtext, f"vscode_assistant:{path.name}", role="assistant")

    def _forward_unique(self, text: str, source: str, role: str):
        payload = (text or "").strip()
        if len(payload) < 10:
            return
        if role == "assistant" and not FORWARD_ASSISTANT:
            return
        # Drop meta/instruction-heavy blocks that pollute graph semantics.
        lowered = payload.lower()
        if "agents.md" in lowered or "my request for codex" in lowered:
            return
        if payload.count("```") >= 2:
            return
        sig = hashlib.sha1((role + "|" + payload).encode("utf-8", errors="ignore")).hexdigest()
        if sig in self._seen:
            return
        self._seen.add(sig)
        if len(self._seen) > 5000:
            # Prevent unbounded growth in long-running sessions.
            self._seen = set(list(self._seen)[-2500:])

        ok = submit_shadow_text(payload, source=source)
        if ok:
            if role == "user":
                self._stats["forwarded_user"] += 1
            else:
                self._stats["forwarded_assistant"] += 1
            self._event("bridge_forward", f"forwarded_{role}", {"source": source, "len": len(payload)})

    def _event(self, kind: str, message: str, extra: Optional[dict] = None):
        BRIDGE_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "message": message,
        }
        if extra:
            row.update(extra)
        with BRIDGE_EVENTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


_BRIDGE: Optional[VSCodeChatBridgeService] = None
_BRIDGE_LOCK = threading.Lock()


def start_vscode_chat_bridge(poll_interval: float = 1.5, lookback_days: int = 30) -> VSCodeChatBridgeService:
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = VSCodeChatBridgeService(poll_interval=poll_interval, lookback_days=lookback_days)
        _BRIDGE.start()
        return _BRIDGE


def stop_vscode_chat_bridge():
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is not None:
            _BRIDGE.stop()


def get_vscode_chat_bridge_status() -> dict:
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            return {
                "running": False,
                "forwarded_user": 0,
                "forwarded_assistant": 0,
                "last_error": "",
                "watched_files": 0,
                "watched_paths": [],
                "started_at": "",
            }
        return _BRIDGE.status()
