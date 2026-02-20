from typing import Dict, List, Tuple


class BackwardVerifier:
    """
    Evidence-first backward checker:
    - Every claim must have either evidence_ids (from retriever/proof) or
      a supporting historical edge / simple rule-support.
    """

    RULE_SUPPORTED_OPS = {"ATTR", "ISA", "CAUSE", "IMPLY", "GOAL", "BEFORE", "OPPOSE", "PREVENT"}

    @staticmethod
    def _norm_instr(instr: dict):
        op = str(instr.get("op", "")).upper().strip()
        args = [str(a) for a in (instr.get("args", []) or [])]
        return op, args

    @staticmethod
    def _history_index(history_ir: List[dict]) -> Dict[Tuple[str, Tuple[str, ...]], int]:
        idx = {}
        for row in history_ir or []:
            op = str(row.get("op", "")).upper().strip()
            args = tuple(str(a) for a in (row.get("args", []) or []))
            if not op or not args:
                continue
            idx[(op, args)] = idx.get((op, args), 0) + 1
        return idx

    @staticmethod
    def _has_evidence(instr: dict) -> bool:
        ev = instr.get("evidence_ids", [])
        return isinstance(ev, list) and any(str(x).strip() for x in ev)

    @staticmethod
    def _rule_support(instr: dict, hist_idx: Dict[Tuple[str, Tuple[str, ...]], int]) -> bool:
        op, args = BackwardVerifier._norm_instr(instr)
        if op == "ATTR" and len(args) >= 3:
            child, _, value = args[0], args[1], args[2]
            for (hop, hargs), _cnt in hist_idx.items():
                if hop == "ISA" and len(hargs) >= 2 and hargs[0] == child:
                    parent = hargs[1]
                    if ("ATTR", (parent, args[1], value)) in hist_idx:
                        return True
        return False

    def verify(self, ir_chain: List[dict], history_ir: List[dict], proof_objects: List[dict] = None):
        hist_idx = self._history_index(history_ir or [])
        unsupported = []
        report = []

        for i, instr in enumerate(ir_chain or []):
            if not isinstance(instr, dict):
                continue
            op, args = self._norm_instr(instr)
            if not op:
                continue

            direct_hist = (op, tuple(args)) in hist_idx
            has_ev = self._has_evidence(instr)
            rule_ok = self._rule_support(instr, hist_idx) if op in self.RULE_SUPPORTED_OPS else False
            supported = bool(has_ev or direct_hist or rule_ok)

            row = {
                "claim_id": f"claim::{i}",
                "op": op,
                "args": args,
                "supported": supported,
                "via_evidence": has_ev,
                "via_history": direct_hist,
                "via_rule": rule_ok,
            }
            report.append(row)
            if not supported:
                unsupported.append(row)

        if unsupported:
            return (
                False,
                f"Backward verifier reject: unsupported_claims={len(unsupported)}/{len(report)}",
                report,
            )
        return True, f"Backward verifier accepted: supported={len(report)}", report

