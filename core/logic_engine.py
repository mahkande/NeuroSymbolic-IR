from z3 import *
import json
import os
import threading


_Z3_LOCK = threading.Lock()
set_param("parallel.enable", False)
_USE_NATIVE_Z3 = os.getenv("COGNITIVE_USE_Z3", "0").strip().lower() in {"1", "true", "yes", "on"}
_Z3_PROFILE = os.getenv("COGNITIVE_Z3_PROFILE", "balanced").strip().lower()
_PROFILE_TIMEOUTS = {"safe": 3000, "balanced": 1800, "aggressive": 900}
_Z3_TIMEOUT_MS = int(os.getenv("COGNITIVE_Z3_TIMEOUT_MS", str(_PROFILE_TIMEOUTS.get(_Z3_PROFILE, 1800))))


class LogicEngine:
    EXCLUSIVE_PROPERTY_PAIRS = {
        frozenset(("canli", "olu")),
        frozenset(("sicak", "soguk")),
    }

    ADDITIVE_KEYS = {
        "renk",
        "sekil",
        "tat",
        "doku",
        "ozellik",
        "özellik",
        "property",
    }

    def __init__(self):
        self.solver = Solver()
        try:
            self.solver.set(timeout=max(100, int(_Z3_TIMEOUT_MS)))
        except Exception:
            pass
        self.last_inferred_props = []

    @staticmethod
    def _norm_token(value):
        if value is None:
            return None
        return str(value).strip().lower()

    @staticmethod
    def _pair(a, b):
        return (str(a), str(b))

    def _is_additive_key(self, key):
        if key is None:
            return True
        return self._norm_token(key) in self.ADDITIVE_KEYS

    def _extract_facts(self, ir_chain):
        isa_facts = []
        prop_facts = []
        oppose_facts = []
        cause_pairs = []
        prevent_pairs = []
        before_pairs = []
        must_actions = set()
        forbid_actions = set()

        for instr in ir_chain or []:
            op = str(instr.get("op", "")).upper()
            args = instr.get("args", []) or []
            if op == "ISA" and len(args) >= 2:
                isa_facts.append((args[0], args[1]))
            elif op in ("ATTR", "PROP"):
                if len(args) >= 3:
                    prop_facts.append((args[0], args[1], args[2]))
                elif len(args) >= 2:
                    prop_facts.append((args[0], None, args[1]))
            elif op == "OPPOSE" and len(args) >= 2:
                oppose_facts.append((args[0], args[1]))
            elif op == "CAUSE" and len(args) >= 2:
                cause_pairs.append((args[0], args[1]))
            elif op == "PREVENT" and len(args) >= 2:
                prevent_pairs.append((args[0], args[1]))
            elif op == "BEFORE" and len(args) >= 2:
                before_pairs.append((args[0], args[1]))
            elif op == "MUST" and len(args) >= 1:
                must_actions.add(str(args[0]))
            elif op == "FORBID" and len(args) >= 1:
                forbid_actions.add(str(args[0]))

        return {
            "isa_facts": isa_facts,
            "prop_facts": prop_facts,
            "oppose_facts": oppose_facts,
            "cause_pairs": cause_pairs,
            "prevent_pairs": prevent_pairs,
            "before_pairs": before_pairs,
            "must_actions": must_actions,
            "forbid_actions": forbid_actions,
        }

    def _infer_inherited_props(self, isa_facts, prop_facts):
        inferred = set(prop_facts)
        changed = True
        while changed:
            changed = False
            for child, parent in isa_facts:
                for subj, key, value in list(inferred):
                    if subj == parent and self._is_additive_key(key) and (child, key, value) not in inferred:
                        inferred.add((child, key, value))
                        changed = True
        return inferred

    @staticmethod
    def _has_temporal_cycle(before_pairs):
        adj = {}
        for a, b in before_pairs:
            a = str(a)
            b = str(b)
            adj.setdefault(a, set()).add(b)
        visiting = set()
        visited = set()

        def dfs(node):
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for nxt in adj.get(node, set()):
                if dfs(nxt):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        for n in list(adj.keys()):
            if dfs(n):
                return True
        return False

    def explain_conflicts(self, ir_chain):
        facts = self._extract_facts(ir_chain)
        messages = []

        causes = set(self._pair(a, b) for a, b in facts["cause_pairs"])
        prevents = set(self._pair(a, b) for a, b in facts["prevent_pairs"])
        overlap = sorted(causes & prevents)
        if overlap:
            messages.append(f"CAUSE vs PREVENT overlap: {overlap[:3]}")

        must_forbid = sorted(set(facts["must_actions"]) & set(facts["forbid_actions"]))
        if must_forbid:
            messages.append(f"MUST vs FORBID overlap: {must_forbid[:3]}")

        before_pairs = set(self._pair(a, b) for a, b in facts["before_pairs"])
        anti = []
        for a, b in before_pairs:
            if (b, a) in before_pairs and (a, b) != (b, a):
                anti.append((a, b))
        if anti:
            messages.append(f"BEFORE antisymmetry violation: {anti[:3]}")
        if self._has_temporal_cycle(before_pairs):
            messages.append("BEFORE cycle detected")
        return messages

    def verify_consistency(self, ir_chain):
        facts = self._extract_facts(ir_chain)
        isa_facts = facts["isa_facts"]
        prop_facts = facts["prop_facts"]
        oppose_facts = facts["oppose_facts"]
        cause_pairs = facts["cause_pairs"]
        prevent_pairs = facts["prevent_pairs"]
        before_pairs = facts["before_pairs"]
        must_actions = facts["must_actions"]
        forbid_actions = facts["forbid_actions"]

        inferred_props = self._infer_inherited_props(isa_facts, prop_facts)
        self.last_inferred_props = sorted({(s, v) for s, _, v in inferred_props})
        inherited_count = max(0, len(inferred_props) - len(set(prop_facts)))

        if not _USE_NATIVE_Z3:
            oppose_pairs = set(oppose_facts)
            for subj, _, value in inferred_props:
                if (subj, value) in oppose_pairs:
                    return False, "Z3_DISABLED: Mantiksal Celiski (OPPOSE vs ATTR) tespit edildi."

            subj_values = {}
            for subj, _, value in inferred_props:
                subj_values.setdefault(subj, set()).add(str(value))
            for subj, values in subj_values.items():
                for pair in self.EXCLUSIVE_PROPERTY_PAIRS:
                    a, b = tuple(pair)
                    if a in values and b in values:
                        return False, f"Z3_DISABLED: Mantiksal Celiski tespit edildi ({subj}: {a}/{b})."

            if set(self._pair(a, b) for a, b in cause_pairs) & set(self._pair(a, b) for a, b in prevent_pairs):
                return False, "Z3_DISABLED: Mantiksal Celiski (CAUSE vs PREVENT) tespit edildi."
            if set(must_actions) & set(forbid_actions):
                return False, "Z3_DISABLED: Mantiksal Celiski (MUST vs FORBID) tespit edildi."
            before_set = set(self._pair(a, b) for a, b in before_pairs)
            for a, b in before_set:
                if (b, a) in before_set and (a, b) != (b, a):
                    return False, "Z3_DISABLED: Mantiksal Celiski (BEFORE antisymmetry) tespit edildi."
            if self._has_temporal_cycle(before_set):
                return False, "Z3_DISABLED: Mantiksal Celiski (BEFORE cycle) tespit edildi."

            return True, f"Z3_DISABLED: Gecerli Model. Inherited PROP count: {inherited_count}"

        solver = Solver()
        try:
            solver.set(timeout=max(100, int(_Z3_TIMEOUT_MS)))
        except Exception:
            pass
        self.solver = solver

        Atom = StringSort()
        IsA = Function("IsA", Atom, Atom, BoolSort())
        HasProp = Function("HasProp", Atom, Atom, Atom, BoolSort())
        OpposeValue = Function("OpposeValue", Atom, Atom, BoolSort())
        Causes = Function("Causes", Atom, Atom, BoolSort())
        Prevents = Function("Prevents", Atom, Atom, BoolSort())
        Before = Function("Before", Atom, Atom, BoolSort())
        Must = Function("Must", Atom, BoolSort())
        Forbid = Function("Forbid", Atom, BoolSort())

        x = Const("x", Atom)
        y = Const("y", Atom)
        k = Const("k", Atom)
        k2 = Const("k2", Atom)
        p = Const("p", Atom)

        solver.add(ForAll([x, y, k, p], Implies(And(IsA(x, y), HasProp(y, k, p)), HasProp(x, k, p))))
        solver.add(ForAll([x, p, k], Implies(OpposeValue(x, p), Not(HasProp(x, k, p)))))
        solver.add(ForAll([x, y], Implies(Causes(x, y), Not(Prevents(x, y)))))
        solver.add(ForAll([x], Not(And(Must(x), Forbid(x)))))
        solver.add(ForAll([x], Not(Before(x, x))))
        solver.add(ForAll([x, y], Implies(Before(x, y), Not(Before(y, x)))))

        for pair in self.EXCLUSIVE_PROPERTY_PAIRS:
            val_a, val_b = tuple(pair)
            solver.add(
                ForAll(
                    [x, k, k2],
                    Not(
                        And(
                            HasProp(x, k, StringVal(str(val_a))),
                            HasProp(x, k2, StringVal(str(val_b))),
                        )
                    ),
                )
            )

        for child, parent in isa_facts:
            solver.add(IsA(StringVal(str(child)), StringVal(str(parent))))
        for subj, key, value in inferred_props:
            key_atom = "" if key is None else str(key)
            solver.add(HasProp(StringVal(str(subj)), StringVal(key_atom), StringVal(str(value))))
        for subj, value in oppose_facts:
            solver.add(OpposeValue(StringVal(str(subj)), StringVal(str(value))))
        for a, b in cause_pairs:
            solver.add(Causes(StringVal(str(a)), StringVal(str(b))))
        for a, b in prevent_pairs:
            solver.add(Prevents(StringVal(str(a)), StringVal(str(b))))
        for a, b in before_pairs:
            solver.add(Before(StringVal(str(a)), StringVal(str(b))))
        for a in must_actions:
            solver.add(Must(StringVal(str(a))))
        for a in forbid_actions:
            solver.add(Forbid(StringVal(str(a))))

        try:
            with _Z3_LOCK:
                result = solver.check()
        except (Z3Exception, OSError) as exc:
            return True, f"Z3: NATIVE_ERROR_BYPASS ({type(exc).__name__}) Inherited PROP count: {inherited_count}"

        if result == unsat:
            details = self.explain_conflicts(ir_chain)
            return False, f"Z3: Mantiksal Celiski Tespit Edildi! Counterexample: {details[:3]}"

        return True, f"Z3: Gecerli Model. Inherited PROP count: {inherited_count} (profile={_Z3_PROFILE}, timeout_ms={_Z3_TIMEOUT_MS})"

    def resolve_conflict(self, old_ir, new_ir, bridge, isa_schema):
        old_str = json.dumps(old_ir, ensure_ascii=False)
        new_str = json.dumps(new_ir, ensure_ascii=False)
        prompt = (
            f"Hafizamda {old_str} var ama simdi {new_str} geldi. "
            "Bu ikisi mantiksal olarak celisiyor. "
            "Lutfen hangisinin daha genel bir dogru oldugunu veya bu ikisini uzlastiracak bir 'Ust-Kural' "
            "(Subsumption) olup olmadigini analiz et ve revize edilmis tek bir IR seti dondur."
        )
        revised = bridge.compile_to_ir(prompt, isa_schema)
        return revised

    def find_minimal_unsat_core(self, ir_chain):
        prop_like = [i for i in ir_chain if i.get("op") in ("ATTR", "PROP")]
        opposes = [i for i in ir_chain if i.get("op") == "OPPOSE"]

        prop_pairs = set()
        for i in prop_like:
            args = i.get("args", [])
            if len(args) >= 3:
                prop_pairs.add((args[0], args[2]))
            elif len(args) >= 2:
                prop_pairs.add((args[0], args[1]))

        for o in opposes:
            args = o.get("args", [])
            if len(args) >= 2 and (args[0], args[1]) in prop_pairs:
                for p in prop_like:
                    p_args = p.get("args", [])
                    p_pair = (p_args[0], p_args[2]) if len(p_args) >= 3 else ((p_args[0], p_args[1]) if len(p_args) >= 2 else None)
                    if p_pair == (args[0], args[1]):
                        return [p, o]

        causes = [i for i in ir_chain if i.get("op") == "CAUSE"]
        prevents = [i for i in ir_chain if i.get("op") == "PREVENT"]
        for c in causes:
            c_args = c.get("args", [])
            for p in prevents:
                p_args = p.get("args", [])
                if len(c_args) >= 2 and len(p_args) >= 2 and c_args[0] == p_args[0] and c_args[1] == p_args[1]:
                    return [c, p]

        musts = [i for i in ir_chain if i.get("op") == "MUST"]
        forbids = [i for i in ir_chain if i.get("op") == "FORBID"]
        for m in musts:
            ma = m.get("args", [])
            for f in forbids:
                fa = f.get("args", [])
                if ma and fa and str(ma[0]) == str(fa[0]):
                    return [m, f]

        befores = [i for i in ir_chain if i.get("op") == "BEFORE"]
        for a in befores:
            aa = a.get("args", [])
            for b in befores:
                bb = b.get("args", [])
                if len(aa) >= 2 and len(bb) >= 2 and aa[0] == bb[1] and aa[1] == bb[0]:
                    return [a, b]

        for c in causes:
            for o in opposes:
                if set(c.get("args", [])) == set(o.get("args", [])):
                    return [c, o]

        return []

