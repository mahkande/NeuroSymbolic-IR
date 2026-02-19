import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from core.logic_engine import LogicEngine
from core.model_bridge import NanbeigeBridge
from core.nlp_utils import grammar_filter_ir, rebalance_relations
from core.parser import IRParser
from core.validator import CognitiveValidator
from main import load_global_graph, save_global_graph
from memory.knowledge_graph import CognitiveMemory


class _ConversationChangeHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_modified(self, event):
        if event.is_directory:
            return
        if Path(event.src_path).resolve() == self.service.conversation_path.resolve():
            self.service.notify_change()

    def on_created(self, event):
        if event.is_directory:
            return
        if Path(event.src_path).resolve() == self.service.conversation_path.resolve():
            self.service.notify_change()


class FileListenerService:
    def __init__(
        self,
        conversation_path: str = "conversation.txt",
        processed_logs_path: str = "memory/processed_logs.txt",
        model_name: str | None = None,
        api_key: str | None = None,
        debounce_sec: float = 0.4,
    ):
        self.conversation_path = Path(conversation_path).resolve()
        self.processed_logs_path = Path(processed_logs_path).resolve()
        self.debounce_sec = max(0.1, float(debounce_sec))

        self.bridge = NanbeigeBridge(model_name=model_name, api_key=api_key)
        self.parser = IRParser()
        self.validator = CognitiveValidator()
        self.logic = LogicEngine()

        self._observer = Observer()
        self._event_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True, name="file-listener-worker")

    def start(self):
        self.conversation_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_logs_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.conversation_path.exists():
            self.conversation_path.write_text("", encoding="utf-8")
        if not self.processed_logs_path.exists():
            self.processed_logs_path.write_text("", encoding="utf-8")

        handler = _ConversationChangeHandler(self)
        self._observer.schedule(handler, str(self.conversation_path.parent), recursive=False)
        self._observer.start()
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        self._event_queue.put("STOP")
        self._observer.stop()
        self._observer.join(timeout=3)
        self._worker.join(timeout=3)

    def notify_change(self):
        self._event_queue.put("CHANGE")

    def _run_worker(self):
        while not self._stop_event.is_set():
            event = self._event_queue.get()
            if event == "STOP":
                break

            # Debounce bursts of file write events.
            time.sleep(self.debounce_sec)
            while True:
                try:
                    nxt = self._event_queue.get_nowait()
                    if nxt == "STOP":
                        return
                except queue.Empty:
                    break

            self._consume_conversation_file()

    def _consume_conversation_file(self):
        try:
            with self.conversation_path.open("r+", encoding="utf-8", errors="ignore") as f:
                payload = f.read()
                f.seek(0)
                f.truncate(0)
        except Exception as exc:
            self._log_processed("listener_error", {"error": str(exc)})
            return

        if not payload.strip():
            return

        blocks = self._split_blocks(payload)
        if not blocks:
            return

        for block in blocks:
            self._process_block(block)
            self._log_processed("consumed_block", {"text": block})

    @staticmethod
    def _split_blocks(text: str) -> List[str]:
        parts = []
        normalized = text.replace("\r", "")
        for part in normalized.split("\n\n"):
            p = part.strip()
            if len(p) >= 8:
                parts.append(p)
        return parts

    @staticmethod
    def _graph_to_ir(graph, known_ops):
        ir = []
        for u, v, attrs in graph.edges(data=True):
            op = attrs.get("relation") or attrs.get("label")
            if op in known_ops:
                ir.append({"op": op, "args": [u, v]})
        return ir

    def _validate_ir_chain(self, ir_chain):
        clean = []
        for instr in ir_chain:
            if not isinstance(instr, dict):
                continue
            op = instr.get("op")
            args = instr.get("args", [])
            if not isinstance(args, list):
                continue
            is_valid, msg = self.validator.validate_instruction(op, args)
            if is_valid:
                clean.append({"op": op, "args": args})
            else:
                self._log_processed("invalid_ir", {"instruction": instr, "reason": msg})
        return clean

    def _process_block(self, block_text: str):
        graph = load_global_graph()
        memory = CognitiveMemory()
        memory.graph = graph

        raw = self.bridge.compile_to_ir(
            block_text,
            self.validator.isa,
            memory_terms=list(graph.nodes),
        )
        if isinstance(raw, dict) and "error" in raw:
            self._log_processed("groq_error", {"text": block_text, "error": raw["error"]})
            return

        parsed = self.parser.parse_raw_output(raw)
        if not isinstance(parsed, list):
            self._log_processed("parse_error", {"text": block_text, "raw": raw})
            return

        ir_chain = self._validate_ir_chain(parsed)
        if not ir_chain:
            self._log_processed("empty_ir", {"text": block_text})
            return

        ir_chain, gf_logs, attr_augment = grammar_filter_ir(ir_chain, known_entities=list(graph.nodes))
        if attr_augment:
            ir_chain.extend(attr_augment)
        ir_chain, rb_logs = rebalance_relations(ir_chain, block_text)
        ir_chain = self._validate_ir_chain(ir_chain)
        if gf_logs or rb_logs:
            self._log_processed("filter_log", {"text": block_text[:200], "logs": (gf_logs + rb_logs)[-8:]})
        if not ir_chain:
            self._log_processed("empty_ir_after_filter", {"text": block_text})
            return

        history_ir = self._graph_to_ir(graph, self.validator.opcodes)
        is_consistent, logic_msg = self.logic.verify_consistency(history_ir + ir_chain)
        if not is_consistent:
            self._log_processed("z3_reject", {"text": block_text, "ir": ir_chain, "reason": logic_msg})
            return

        conflict, conflict_msg = memory.find_conflicts_with_history(ir_chain)
        if conflict:
            self._log_processed("history_reject", {"text": block_text, "ir": ir_chain, "reason": conflict_msg})
            return

        memory.add_ir_chain(ir_chain)
        save_global_graph(memory.graph)
        self._log_processed("graph_write", {"text": block_text, "ir": ir_chain, "z3": logic_msg})

    def _log_processed(self, kind: str, payload: dict):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": payload,
        }
        with self.processed_logs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_file_listener(
    conversation_path: str = "conversation.txt",
    processed_logs_path: str = "memory/processed_logs.txt",
    model_name: str | None = None,
    api_key: str | None = None,
    debounce_sec: float = 0.4,
):
    service = FileListenerService(
        conversation_path=conversation_path,
        processed_logs_path=processed_logs_path,
        model_name=model_name,
        api_key=api_key,
        debounce_sec=debounce_sec,
    )
    service.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
