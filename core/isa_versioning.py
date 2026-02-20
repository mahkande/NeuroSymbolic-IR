import json
from pathlib import Path
from typing import Dict, List, Tuple


REGISTRY_PATH = Path("spec/isa_registry.json")


def _parse_version(v: str) -> Tuple[int, ...]:
    parts = str(v).strip().split(".")
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            out.append(0)
    return tuple(out or [0])


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        return {
            "active_version": "1.0",
            "versions": {"1.0": {"path": "spec/isa_v1.json", "status": "active"}},
            "migration_policy": {"strict_opcode": True, "strict_args": True},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(data: dict, path: Path = REGISTRY_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_versions(path: Path = REGISTRY_PATH) -> List[str]:
    reg = load_registry(path)
    versions = list((reg.get("versions") or {}).keys())
    versions.sort(key=_parse_version)
    return versions


def get_active_version(path: Path = REGISTRY_PATH) -> str:
    reg = load_registry(path)
    active = str(reg.get("active_version") or "").strip()
    if active:
        return active
    versions = list_versions(path)
    return versions[-1] if versions else "1.0"


def get_active_isa_path(path: Path = REGISTRY_PATH) -> str:
    reg = load_registry(path)
    active = get_active_version(path)
    versions = reg.get("versions") or {}
    entry = versions.get(active) or {}
    return str(entry.get("path") or "spec/isa_v1.json")


def set_active_version(version: str, path: Path = REGISTRY_PATH):
    reg = load_registry(path)
    versions = reg.get("versions") or {}
    if version not in versions:
        raise ValueError(f"Unknown ISA version: {version}")
    reg["active_version"] = version
    save_registry(reg, path)


def migrate_instruction(instr: dict, from_version: str, to_version: str) -> dict:
    if not isinstance(instr, dict):
        return instr
    op = str(instr.get("op", ""))
    args = list(instr.get("args", []))

    # 1.0 -> 1.1 migration policy (normalization-safe transform)
    if _parse_version(from_version) <= (1, 0) and _parse_version(to_version) >= (1, 1):
        if op == "ATTR" and len(args) >= 2 and str(args[1]).strip() in {"özellik", "property"}:
            args[1] = "ozellik"
        if op == "BELIEVE" and len(args) >= 3:
            try:
                c = float(args[2])
            except Exception:
                c = 0.5
            c = min(1.0, max(0.0, c))
            args[2] = f"{c:.2f}"
    return {"op": op, "args": args}


def migrate_ir_chain(ir_chain: List[dict], from_version: str, to_version: str) -> List[dict]:
    if from_version == to_version:
        return list(ir_chain or [])
    out = []
    for instr in ir_chain or []:
        out.append(migrate_instruction(instr, from_version, to_version))
    return out
