import asyncio
import hashlib
import json
import os
import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from core.logic_engine import LogicEngine
from core.fallback_rules import merge_fallback, semantic_fallback_ir
from core.hybrid_semantics import RuleBasedExtractorV2
from core.model_bridge import NanbeigeBridge
from core.nlp_utils import grammar_filter_ir, rebalance_relations
from core.parser import IRParser
from core.rule_guard import apply_active_rules, auto_review_candidates, register_unknown_pattern
from core.rule_learner import propose_candidates_from_text, propose_llm_rule_candidate
from core.quality_metrics import record_fallback_usage
from core.validator import CognitiveValidator
from memory.knowledge_graph import CognitiveMemory


SHADOW_EVENTS_PATH = Path("memory/shadow_events.jsonl")
SHADOW_INBOX_PATH = Path("memory/shadow_inbox.txt")
SHADOW_CONVERSATION_PATH = Path("conversation.txt")
SHADOW_ALLOW_ASSISTANT = os.getenv("SHADOW_ALLOW_ASSISTANT", "1").strip().lower() in {"1", "true", "yes", "on"}

SYSTEM_PROMPT = (
    "Sen bir mantik noterisin. Kullanicı ile Code AI arasindaki bu teknik konusmayi izle. "
    "Sadece kalici mantiksal kurallari, tanimlari ve iliski turlerini bizim ISA.json formatimizda "
    "IR'lara donustur. Sohbet kelimelerini (merhaba, tesekkurler vb.) filtrele. "
    "Yanit sadece JSON listesi olsun; bos ise [] dondur."
)


@dataclass
class ShadowChunk:
    source: str
    content: str


class ShadowListenerService:
    def __init__(self, watch_files: Optional[List[str]] = None, poll_interval: float = 1.5, model: Optional[str] = None):
        self.watch_files = self._resolve_watch_files(watch_files)
        self._snapshot_files = self._resolve_snapshot_files()
        self.poll_interval = max(0.5, float(poll_interval))
        self.model = model or os.getenv("GROQ_MODEL") or NanbeigeBridge.DEFAULT_MODEL
        self.api_key = os.getenv("GROQ_API_KEY") or NanbeigeBridge.DEFAULT_API_KEY

        self.validator = CognitiveValidator()
        self.parser = IRParser()

        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._queue = None
        self._loop = None

        self._file_offsets: Dict[str, int] = {}
        self._file_snapshots: Dict[str, str] = {}
        self._recent_hashes = deque(maxlen=512)
        self._clipboard_last = ""
        self._stats = {
            "processed_blocks": 0,
            "produced_ir": 0,
            "rejected_ir": 0,
            "last_error": "",
            "started_at": "",
        }

    def _resolve_watch_files(self, watch_files: Optional[List[str]]) -> List[str]:
        env_files = os.getenv("SHADOW_WATCH_FILES", "")
        paths = []
        if watch_files:
            paths.extend(watch_files)
        if env_files:
            paths.extend([p.strip() for p in env_files.split(";") if p.strip()])

        if not paths:
            paths = [
                str(SHADOW_INBOX_PATH),
                str(SHADOW_CONVERSATION_PATH),
                ".cursor/chat.log",
                ".vscode/chat.log",
                "logs/code_ai_chat.log",
                "logs/terminal_output.log",
            ]

        resolved = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = Path.cwd() / pp
            resolved.append(str(pp))

        # Ensure inbox exists for easy manual feed/paste fallback.
        SHADOW_INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not SHADOW_INBOX_PATH.exists():
            SHADOW_INBOX_PATH.write_text("", encoding="utf-8")
        if not SHADOW_CONVERSATION_PATH.exists():
            SHADOW_CONVERSATION_PATH.write_text("", encoding="utf-8")

        return list(dict.fromkeys(resolved))

    def _resolve_snapshot_files(self) -> set:
        env_files = os.getenv("SHADOW_SNAPSHOT_FILES", "")
        paths = [str((Path.cwd() / SHADOW_CONVERSATION_PATH).resolve())]
        if env_files:
            for p in [x.strip() for x in env_files.split(";") if x.strip()]:
                pp = Path(p)
                if not pp.is_absolute():
                    pp = Path.cwd() / pp
                paths.append(str(pp.resolve()))
        return set(paths)

    def start(self):
        with self._lock:
            if self._running:
                return self
            self._running = True
            self._stats["started_at"] = datetime.now(timezone.utc).isoformat()
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="shadow-listener")
            self._thread.start()
        return self

    def stop(self):
        with self._lock:
            self._running = False
            if self._loop is not None and self._queue is not None:
                try:
                    self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
                except Exception:
                    pass

    def is_running(self):
        return bool(self._running and self._thread and self._thread.is_alive())

    def status(self):
        return {
            "running": self.is_running(),
            "watch_files": self.watch_files,
            **self._stats,
        }

    def _run_loop(self):
        asyncio.run(self._main())

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        producer = asyncio.create_task(self._produce_chunks())
        consumer = asyncio.create_task(self._consume_chunks())
        await asyncio.gather(producer, consumer)

    async def _produce_chunks(self):
        while self._running:
            try:
                for chunk in self._poll_file_chunks():
                    await self._queue.put(chunk)

                clipboard_chunk = self._poll_clipboard_chunk()
                if clipboard_chunk is not None:
                    await self._queue.put(clipboard_chunk)
            except Exception as exc:
                self._stats["last_error"] = str(exc)
                self._append_event(
                    {
                        "type": "listener_error",
                        "level": "error",
                        "message": f"[DASHBOARD_ALERT] Listener producer error: {exc}",
                    }
                )
            await asyncio.sleep(self.poll_interval)

    def _poll_file_chunks(self) -> List[ShadowChunk]:
        chunks: List[ShadowChunk] = []
        for path_str in self.watch_files:
            path = Path(path_str)
            if not path.exists() or not path.is_file():
                continue

            # Inbox mode: consume full content, then truncate for next cycle.
            if path.resolve() == SHADOW_INBOX_PATH.resolve():
                try:
                    payload = path.read_text(encoding="utf-8", errors="ignore")
                    if payload.strip():
                        for block in self._chunk_text(payload):
                            chunks.append(ShadowChunk(source=f"file:{path.name}", content=block))
                        path.write_text("", encoding="utf-8")
                        self._file_offsets[path_str] = 0
                except Exception:
                    pass
                continue

            # Snapshot mode: process on content change (save/overwrite friendly).
            if str(path.resolve()) in self._snapshot_files:
                try:
                    payload = path.read_text(encoding="utf-8", errors="ignore")
                    digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()
                    prev_digest = self._file_snapshots.get(path_str)
                    self._file_snapshots[path_str] = digest
                    if digest == prev_digest or not payload.strip():
                        continue
                    for block in self._chunk_text(payload):
                        chunks.append(ShadowChunk(source=f"file:{path.name}", content=block))
                except Exception:
                    pass
                continue

            prev_offset = self._file_offsets.get(path_str, 0)
            try:
                size = path.stat().st_size
                if prev_offset > size:
                    prev_offset = 0

                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    f.seek(prev_offset)
                    appended = f.read()
                    self._file_offsets[path_str] = f.tell()

                if not appended.strip():
                    continue

                for block in self._chunk_text(appended):
                    chunks.append(ShadowChunk(source=f"file:{path.name}", content=block))
            except Exception:
                continue
        return chunks

    def _poll_clipboard_chunk(self) -> Optional[ShadowChunk]:
        # Optional fallback: read clipboard when pyperclip is available.
        try:
            import pyperclip  # type: ignore
        except Exception:
            return None

        try:
            text = (pyperclip.paste() or "").strip()
        except Exception:
            return None

        if not text or text == self._clipboard_last:
            return None
        if len(text) < 20:
            return None

        self._clipboard_last = text
        return ShadowChunk(source="clipboard", content=text)

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        blocks = []
        raw = text.replace("\r", "")
        for part in raw.split("\n\n"):
            p = part.strip()
            if len(p) >= 20:
                blocks.append(p)
        return blocks

    async def _consume_chunks(self):
        try:
            from groq import Groq
        except Exception as exc:
            self._stats["last_error"] = str(exc)
            self._append_event(
                {
                    "type": "shadow_error",
                    "level": "error",
                    "message": "[DASHBOARD_ALERT] groq paketi kurulu degil. Lutfen 'pip install groq' calistirin.",
                }
            )
            return

        client = Groq(api_key=self.api_key, timeout=5.0)

        while self._running:
            chunk = await self._queue.get()
            if chunk is None and not self._running:
                break
            if chunk is None:
                continue
            if chunk.source.startswith("codex_assistant:") and not SHADOW_ALLOW_ASSISTANT:
                continue

            fingerprint = hashlib.sha1(chunk.content.encode("utf-8", errors="ignore")).hexdigest()
            if fingerprint in self._recent_hashes:
                continue
            self._recent_hashes.append(fingerprint)

            self._stats["processed_blocks"] += 1
            try:
                llm_output = await self._dispatch_to_groq(client, chunk.content)
                ir_chain = self.parser.parse_raw_output(
                    llm_output,
                    allowed_ops=set(self.validator.opcodes.keys()),
                    strict_schema=True,
                )
                used_learned_rule = False
                used_fallback = False
                if not isinstance(ir_chain, list):
                    # Rule gate: try only approved learned rules first.
                    ir_chain = apply_active_rules(chunk.content, self.validator)
                    used_learned_rule = bool(ir_chain)
                    if not ir_chain:
                        # Fallback: use deterministic extractor when LLM output is not parseable.
                        ir_chain = self._rule_based_ir(chunk.content)
                        used_fallback = bool(ir_chain)
                        if used_fallback:
                            record_fallback_usage("shadow_listener", chunk.content, ir_chain, reason="parse_failed")
                    if not ir_chain:
                        register_unknown_pattern(chunk.content, source=chunk.source, reason="parse_failed")
                        self._append_event(
                            {
                                "type": "shadow_parse",
                                "level": "warn",
                                "source": chunk.source,
                                "text": chunk.content[:500],
                                "message": "LLM output parse edilemedi.",
                            }
                        )
                        continue

                clean_ir = self._sanitize_ir(ir_chain)
                if not clean_ir:
                    clean_ir = apply_active_rules(chunk.content, self.validator)
                    if clean_ir:
                        used_learned_rule = True
                    if not clean_ir:
                        clean_ir = self._rule_based_ir(chunk.content)
                        used_fallback = bool(clean_ir)
                        if used_fallback:
                            record_fallback_usage("shadow_listener", chunk.content, clean_ir, reason="sanitize_empty")
                if not clean_ir:
                    register_unknown_pattern(chunk.content, source=chunk.source, reason="empty_ir")
                    self._append_event(
                        {
                            "type": "shadow_skip",
                            "level": "info",
                            "source": chunk.source,
                            "text": chunk.content[:500],
                            "message": "Kalici mantik IR bulunamadi.",
                        }
                    )
                    continue

                clean_ir, gf_logs, attr_augment = grammar_filter_ir(clean_ir, known_entities=[])
                if attr_augment:
                    clean_ir.extend(attr_augment)
                clean_ir, rb_logs = rebalance_relations(clean_ir, chunk.content)
                clean_ir = self._sanitize_ir(clean_ir)
                if not clean_ir:
                    register_unknown_pattern(chunk.content, source=chunk.source, reason="empty_ir_after_filter")
                    proposal = propose_llm_rule_candidate(
                        chunk.content,
                        self.validator,
                        provider="groq",
                        model_name=self.model,
                        api_key=self.api_key,
                        source="listener_llm_rule_bootstrap",
                    )
                    if proposal:
                        cand = proposal.get("candidate") or {}
                        q = str(proposal.get("question", "")).strip()
                        conf = float(proposal.get("confidence", 0.0) or 0.0)
                        msg = f"LLM rule adayi olusturuldu id={cand.get('id', 'n/a')} conf={conf:.2f}"
                        if q:
                            msg += f" | soru: {q}"
                        self._append_event(
                            {
                                "type": "rule_bootstrap",
                                "level": "info",
                                "source": chunk.source,
                                "text": chunk.content[:500],
                                "message": msg[:500],
                            }
                        )
                    continue

                if self._is_do_only(clean_ir):
                    register_unknown_pattern(chunk.content, source=chunk.source, reason="do_only_fallback")
                    proposal = propose_llm_rule_candidate(
                        chunk.content,
                        self.validator,
                        provider="groq",
                        model_name=self.model,
                        api_key=self.api_key,
                        source="listener_llm_rule_bootstrap",
                    )
                    if proposal:
                        cand = proposal.get("candidate") or {}
                        q = str(proposal.get("question", "")).strip()
                        conf = float(proposal.get("confidence", 0.0) or 0.0)
                        cid = cand.get("id", "n/a")
                        msg = f"LLM rule adayi olusturuldu id={cid} conf={conf:.2f}"
                        if q:
                            msg += f" | soru: {q}"
                        self._append_event(
                            {
                                "type": "rule_bootstrap",
                                "level": "info",
                                "source": chunk.source,
                                "text": chunk.content[:500],
                                "message": msg[:500],
                            }
                        )
                        if proposal.get("auto_apply"):
                            cand_ir = self._sanitize_ir(proposal.get("ir", []))
                            if cand_ir and not self._is_do_only(cand_ir):
                                clean_ir = cand_ir
                                used_fallback = True
                                record_fallback_usage("shadow_listener", chunk.content, clean_ir, reason="do_only_bootstrap")
                if gf_logs or rb_logs:
                    self._append_event(
                        {
                            "type": "shadow_filter",
                            "level": "info",
                            "source": chunk.source,
                            "message": "; ".join((gf_logs + rb_logs)[-6:])[:500],
                        }
                    )

                if used_fallback:
                    propose_candidates_from_text(chunk.content, clean_ir, source="listener_fallback")
                    auto_review_candidates(self.validator, min_hits=3)

                self._apply_shadow_ir(clean_ir, chunk)
            except Exception as exc:
                self._stats["last_error"] = str(exc)
                mapped = self._map_groq_error(exc)
                self._append_event(
                    {
                        "type": "shadow_error",
                        "level": "error",
                        "source": chunk.source,
                        "text": chunk.content[:500],
                        "message": mapped,
                    }
                )

    @staticmethod
    def _dispatch_sync(client, model: str, text_block: str):
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_block},
            ],
            temperature=0.0,
            max_tokens=300,
        )

    async def _dispatch_to_groq(self, client, text_block: str) -> str:
        completion = await asyncio.wait_for(
            asyncio.to_thread(self._dispatch_sync, client, self.model, text_block),
            timeout=8.0,
        )
        return completion.choices[0].message.content if completion.choices else "[]"

    def _sanitize_ir(self, ir_chain: List[dict]) -> List[dict]:
        clean = []
        for instr in ir_chain:
            if not isinstance(instr, dict):
                continue
            op = instr.get("op")
            args = instr.get("args", [])
            if op not in self.validator.opcodes:
                continue
            if not isinstance(args, list):
                continue
            is_valid, _ = self.validator.validate_instruction(op, args)
            if not is_valid:
                continue
            clean.append({"op": op, "args": args})
        return clean

    @staticmethod
    def _norm_token(text: str) -> str:
        token = (text or "").strip().lower()
        token = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ-]", " ", token)
        token = re.sub(r"\s+", " ", token).strip()
        token = token.replace(" ", "_")
        return token

    def _rule_based_ir(self, text: str) -> List[dict]:
        extractor = RuleBasedExtractorV2()
        rule_ir = extractor.extract_ir(text)
        semantic_ir = semantic_fallback_ir(text, include_do=True)
        merged = merge_fallback(rule_ir, semantic_ir)
        return self._sanitize_ir(merged)

    @staticmethod
    def _is_do_only(ir_chain: List[dict]) -> bool:
        rows = [r for r in (ir_chain or []) if isinstance(r, dict)]
        if not rows:
            return False
        return all(str(r.get("op", "")).upper() == "DO" for r in rows)

    @staticmethod
    def _graph_to_ir(graph, known_ops: Dict[str, dict]) -> List[dict]:
        ir = []
        for u, v, attrs in graph.edges(data=True):
            op = attrs.get("relation") or attrs.get("label")
            if op in known_ops:
                ir.append({"op": op, "args": [u, v]})
        return ir

    def _apply_shadow_ir(self, ir_chain: List[dict], chunk: ShadowChunk):
        from main import load_global_graph, save_global_graph

        with self._lock:
            graph = load_global_graph()
            memory = CognitiveMemory()
            memory.graph = graph

            history_ir = self._graph_to_ir(graph, self.validator.opcodes)
            z3 = LogicEngine()
            is_consistent, logic_msg = z3.verify_consistency(history_ir + ir_chain)

            if not is_consistent:
                self._stats["rejected_ir"] += 1
                self._append_event(
                    {
                        "type": "shadow_ir",
                        "level": "warn",
                        "source": chunk.source,
                        "text": chunk.content[:500],
                        "ir": ir_chain,
                        "z3_status": logic_msg,
                        "message": f"[DASHBOARD_ALERT] Shadow IR reddedildi: {logic_msg}",
                    }
                )
                return

            conflict, conflict_msg = memory.find_conflicts_with_history(ir_chain)
            if conflict:
                self._stats["rejected_ir"] += 1
                self._append_event(
                    {
                        "type": "shadow_ir",
                        "level": "warn",
                        "source": chunk.source,
                        "text": chunk.content[:500],
                        "ir": ir_chain,
                        "z3_status": "UNSAT",
                        "message": f"[DASHBOARD_ALERT] Shadow IR tarihsel celiski: {conflict_msg}",
                    }
                )
                return

            memory.add_ir_chain(ir_chain)
            save_global_graph(memory.graph)

            self._stats["produced_ir"] += len(ir_chain)
            self._append_event(
                {
                    "type": "shadow_ir",
                    "level": "info",
                    "source": chunk.source,
                    "text": chunk.content[:500],
                    "ir": ir_chain,
                    "z3_status": logic_msg,
                    "message": "Shadow IR hafizaya yazildi.",
                }
            )

    @staticmethod
    def _map_groq_error(exc: Exception) -> str:
        msg = str(exc)
        lmsg = msg.lower()
        status = getattr(exc, "status_code", None)

        if status == 429 or "quota" in lmsg or "rate limit" in lmsg or "insufficient_quota" in lmsg:
            return "[DASHBOARD_ALERT] Groq kotasi/hiz limiti doldu. Dashboard: https://console.groq.com"

        network_tokens = ["connection", "timeout", "network", "dns", "service unavailable", "reset", "refused"]
        if any(tok in lmsg for tok in network_tokens):
            return "[DASHBOARD_ALERT] Groq baglantisi koptu. Ag ve Dashboard'u kontrol edin: https://console.groq.com"

        return f"[DASHBOARD_ALERT] Shadow dispatcher hatasi: {msg}"

    def _append_event(self, event: dict):
        SHADOW_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with SHADOW_EVENTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_recent_shadow_events(limit: int = 30) -> List[dict]:
    if not SHADOW_EVENTS_PATH.exists():
        return []
    lines = SHADOW_EVENTS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


_SERVICE: Optional[ShadowListenerService] = None
_SERVICE_LOCK = threading.Lock()


def start_shadow_listener(
    watch_files: Optional[List[str]] = None,
    poll_interval: float = 1.5,
    model: Optional[str] = None,
    force_restart: bool = False,
) -> ShadowListenerService:
    global _SERVICE
    with _SERVICE_LOCK:
        if force_restart and _SERVICE is not None:
            _SERVICE.stop()
            _SERVICE = None
        if _SERVICE is None:
            _SERVICE = ShadowListenerService(watch_files=watch_files, poll_interval=poll_interval, model=model)
        _SERVICE.start()
        return _SERVICE


def stop_shadow_listener():
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is not None:
            _SERVICE.stop()


def get_shadow_listener_status() -> dict:
    with _SERVICE_LOCK:
        if _SERVICE is None:
            return {
                "running": False,
                "watch_files": [],
                "processed_blocks": 0,
                "produced_ir": 0,
                "rejected_ir": 0,
                "last_error": "",
                "started_at": "",
            }
        return _SERVICE.status()


def submit_shadow_text(text: str, source: str = "chat_bridge") -> bool:
    payload = (text or "").strip()
    if len(payload) < 5:
        return False

    service = start_shadow_listener()
    chunk = ShadowChunk(source=source, content=payload)

    with _SERVICE_LOCK:
        if service._loop is not None and service._queue is not None and service.is_running():
            try:
                service._loop.call_soon_threadsafe(service._queue.put_nowait, chunk)
                return True
            except Exception:
                pass

    # Fallback path if queue is not available yet.
    SHADOW_INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SHADOW_INBOX_PATH.open("a", encoding="utf-8") as f:
        f.write(payload + "\n\n")
    return True

