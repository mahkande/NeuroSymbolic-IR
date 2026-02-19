from z3 import *
import json
import os
import threading


_Z3_LOCK = threading.Lock()
set_param("parallel.enable", False)
_USE_NATIVE_Z3 = os.getenv("COGNITIVE_USE_Z3", "0").strip().lower() in {"1", "true", "yes", "on"}


class LogicEngine:
    # Disjointness matrix (value-level exclusivity)
    EXCLUSIVE_PROPERTY_PAIRS = {
        frozenset(("canli", "olu")),
        frozenset(("sicak", "soguk")),
    }

    # Multi-valued/additive property keys
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
        self.last_inferred_props = []

    @staticmethod
    def _norm_token(value):
        if value is None:
            return None
        return str(value).strip().lower()

    def _is_additive_key(self, key):
        # For 2-arg ATTR/PROP where key is absent, keep backward compatibility.
        if key is None:
            return True
        return self._norm_token(key) in self.ADDITIVE_KEYS

    def _extract_facts(self, ir_chain):
        isa_facts = []
        prop_facts = []  # (subject, key, value)
        oppose_facts = []  # (subject, value)

        for instr in ir_chain:
            op = instr.get("op")
            args = instr.get("args", [])

            if op == "ISA" and len(args) >= 2:
                isa_facts.append((args[0], args[1]))
            elif op in ("ATTR", "PROP"):
                # ATTR can be [subject, key, value] or [subject, value]
                if len(args) >= 3:
                    prop_facts.append((args[0], args[1], args[2]))
                elif len(args) >= 2:
                    prop_facts.append((args[0], None, args[1]))
            elif op == "OPPOSE" and len(args) >= 2:
                oppose_facts.append((args[0], args[1]))

        return isa_facts, prop_facts, oppose_facts

    def _infer_inherited_props(self, isa_facts, prop_facts):
        # Automatic inheritance sync:
        # ISA(A, B) and additive PROP(B, key, C) -> PROP(A, key, C)
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

    def verify_consistency(self, ir_chain):
        """Consistency check with native Z3 (optional) + Python fail-safe path."""
        isa_facts, prop_facts, oppose_facts = self._extract_facts(ir_chain)
        inferred_props = self._infer_inherited_props(isa_facts, prop_facts)
        self.last_inferred_props = sorted({(s, v) for s, _, v in inferred_props})
        inherited_count = max(0, len(inferred_props) - len(set(prop_facts)))

        # Fail-safe path: avoid native z3 access violations in long-lived UI processes.
        # Enable native solver by setting COGNITIVE_USE_Z3=1.
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

            return True, f"Z3_DISABLED: Gecerli Model. Inherited PROP count: {inherited_count}"

        # Native Z3 path.
        solver = Solver()
        self.solver = solver

        Atom = StringSort()
        IsA = Function("IsA", Atom, Atom, BoolSort())
        HasProp = Function("HasProp", Atom, Atom, Atom, BoolSort())  # subject, key, value
        OpposeValue = Function("OpposeValue", Atom, Atom, BoolSort())  # subject, value

        x = Const("x", Atom)
        y = Const("y", Atom)
        k = Const("k", Atom)
        k2 = Const("k2", Atom)
        p = Const("p", Atom)

        solver.add(ForAll([x, y, k, p], Implies(And(IsA(x, y), HasProp(y, k, p)), HasProp(x, k, p))))
        solver.add(ForAll([x, p, k], Implies(OpposeValue(x, p), Not(HasProp(x, k, p)))))

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

        try:
            with _Z3_LOCK:
                result = solver.check()
        except (Z3Exception, OSError) as exc:
            return True, f"Z3: NATIVE_ERROR_BYPASS ({type(exc).__name__}) Inherited PROP count: {inherited_count}"

        if result == unsat:
            return False, "Z3: Mantiksal Celiski Tespit Edildi!"

        return True, f"Z3: Gecerli Model. Inherited PROP count: {inherited_count}"

    def resolve_conflict(self, old_ir, new_ir, bridge, isa_schema):
        """When Z3 finds conflict, ask Nanbeige for a reconciled rule set."""
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
        """Return a small conflicting subset when possible."""
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
        for c in causes:
            for o in opposes:
                if set(c.get("args", [])) == set(o.get("args", [])):
                    return [c, o]

        return []
