from typing import Dict, List, Tuple


class DeterministicVerifierGate:
    """
    Final deterministic gate for LLM/hybrid IR before memory write.
    """

    POLICY_PROFILES: Dict[str, Dict[str, float]] = {
        "safe": {
            "min_confidence": 0.65,
            "max_conflict_risk": 0.35,
            "require_provenance": 1.0,
            "escalate_risk": 0.22,
        },
        "balanced": {
            "min_confidence": 0.50,
            "max_conflict_risk": 0.55,
            "require_provenance": 0.0,
            "escalate_risk": 0.35,
        },
        "aggressive": {
            "min_confidence": 0.35,
            "max_conflict_risk": 0.80,
            "require_provenance": 0.0,
            "escalate_risk": 0.55,
        },
    }

    def __init__(self, validator, logic_engine, profile: str = "balanced", strict_mode: bool = False):
        self.validator = validator
        self.logic_engine = logic_engine
        self.profile_name = (profile or "balanced").strip().lower()
        if self.profile_name not in self.POLICY_PROFILES:
            self.profile_name = "balanced"
        self.strict_mode = bool(strict_mode or self.profile_name == "safe")
        self.policy = dict(self.POLICY_PROFILES[self.profile_name])
        if self.strict_mode:
            self.policy["min_confidence"] = max(0.65, float(self.policy["min_confidence"]))
            self.policy["require_provenance"] = 1.0
            self.policy["max_conflict_risk"] = min(float(self.policy["max_conflict_risk"]), 0.45)
            self.policy["escalate_risk"] = min(float(self.policy["escalate_risk"]), 0.30)

    @staticmethod
    def _normalize(ir_chain: List[dict]) -> List[dict]:
        out = []
        for instr in ir_chain or []:
            if not isinstance(instr, dict):
                continue
            op = instr.get("op")
            args = instr.get("args", [])
            if not isinstance(args, list):
                continue
            payload = {"op": op, "args": [str(a) for a in args]}
            if "confidence" in instr:
                payload["confidence"] = instr.get("confidence")
            if "provenance" in instr:
                payload["provenance"] = instr.get("provenance")
            if "source" in instr:
                payload["source"] = instr.get("source")
            out.append(payload)
        return out

    @staticmethod
    def _to_float(val, default=0.0):
        try:
            return float(val)
        except Exception:
            return float(default)

    def _policy_checks(self, checked: List[dict]) -> Tuple[bool, str]:
        min_conf = float(self.policy.get("min_confidence", 0.0))
        require_prov = bool(self.policy.get("require_provenance", 0.0) >= 1.0)
        for instr in checked:
            conf = self._to_float(instr.get("confidence"), default=0.66)
            if conf < min_conf:
                return False, f"Verifier: low-confidence IR rejected (op={instr.get('op')}, conf={conf:.2f}, min={min_conf:.2f})"
            if require_prov:
                prov = str(instr.get("provenance", "")).strip()
                src = str(instr.get("source", "")).strip()
                if not prov and not src:
                    return False, f"Verifier: strict mode requires provenance/source (op={instr.get('op')})"
        return True, "ok"

    @staticmethod
    def _collect_pairs(ir_chain: List[dict], op: str) -> set:
        target = str(op or "").upper()
        out = set()
        for instr in ir_chain or []:
            iop = str(instr.get("op", "")).upper()
            args = instr.get("args", [])
            if iop != target or len(args) < 2:
                continue
            out.add((str(args[0]), str(args[1])))
        return out

    def _conflict_risk_score(self, history_ir: List[dict], checked: List[dict]) -> Tuple[float, List[str]]:
        notes: List[str] = []
        risk = 0.0

        hist_isa = self._collect_pairs(history_ir, "ISA")
        hist_oppose = {tuple(sorted(x)) for x in self._collect_pairs(history_ir, "OPPOSE")}
        new_isa = self._collect_pairs(checked, "ISA")
        new_oppose = {tuple(sorted(x)) for x in self._collect_pairs(checked, "OPPOSE")}

        overlap_conflicts = 0
        for a, b in new_isa:
            if tuple(sorted((a, b))) in hist_oppose:
                overlap_conflicts += 1
        for a, b in new_oppose:
            if (a, b) in hist_isa or (b, a) in hist_isa:
                overlap_conflicts += 1
        if overlap_conflicts:
            risk += min(0.7, 0.25 * overlap_conflicts)
            notes.append(f"history_conflicts={overlap_conflicts}")

        # Duplicate pressure (same op+args repeated in incoming chain).
        seen = set()
        dup = 0
        for instr in checked:
            sig = (str(instr.get("op", "")).upper(), tuple(str(a) for a in instr.get("args", [])))
            if sig in seen:
                dup += 1
            seen.add(sig)
        if dup:
            risk += min(0.2, dup * 0.05)
            notes.append(f"duplicates={dup}")

        low_conf = sum(1 for i in checked if self._to_float(i.get("confidence"), 0.66) < 0.55)
        if low_conf:
            risk += min(0.25, low_conf * 0.04)
            notes.append(f"low_conf_count={low_conf}")

        return round(min(1.0, risk), 3), notes

    def verify(self, ir_chain: List[dict], history_ir: List[dict] = None) -> Tuple[bool, str, List[dict]]:
        history_ir = history_ir or []
        clean = self._normalize(ir_chain)
        if not clean:
            return False, "Verifier: empty IR", []

        checked = []
        for instr in clean:
            is_valid, msg = self.validator.validate_instruction(instr.get("op"), instr.get("args", []))
            if not is_valid:
                return False, f"Verifier: invalid instruction -> {msg}", []
            checked.append(instr)

        policy_ok, policy_msg = self._policy_checks(checked)
        if not policy_ok:
            return False, policy_msg, []

        risk, notes = self._conflict_risk_score(history_ir, checked)
        max_risk = float(self.policy.get("max_conflict_risk", 1.0))
        esc_risk = float(self.policy.get("escalate_risk", max_risk))
        if risk > max_risk:
            return (
                False,
                f"Verifier: conflict-risk reject (risk={risk:.2f}, max={max_risk:.2f}, profile={self.profile_name}, notes={';'.join(notes) or 'none'})",
                [],
            )
        if risk > esc_risk:
            return (
                False,
                f"Verifier: conflict-risk escalate (risk={risk:.2f}, escalate={esc_risk:.2f}, profile={self.profile_name}, notes={';'.join(notes) or 'none'})",
                [],
            )

        is_ok, logic_msg = self.logic_engine.verify_consistency(history_ir + checked)
        if not is_ok:
            return False, f"Verifier: rejected by deterministic logic ({logic_msg})", []
        return True, f"Verifier: accepted ({logic_msg}; profile={self.profile_name}; risk={risk:.2f})", checked
