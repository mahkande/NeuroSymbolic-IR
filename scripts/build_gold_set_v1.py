#!/usr/bin/env python
import argparse
import json
from pathlib import Path


SUBJECTS = [
    "ali", "ayse", "mehmet", "zeynep", "murat", "selin", "can", "deniz", "emre", "eda",
    "yazilim", "sistem", "model", "robot", "sensor", "veri", "hava", "ruzgar", "yagmur", "isik",
]
OBJECTS = [
    "canli", "arac", "cihaz", "surec", "kaynak", "plan", "hedef", "kural", "durum", "olay",
    "proje", "servis", "ag", "platform", "sinyal", "enerji", "kalite", "performans", "zaman", "risk",
]
VERBS = [
    "calis", "ogren", "yaz", "test", "olc", "izle", "kaydet", "planla", "tasarla", "guncelle",
    "koru", "denetle", "onayla", "temizle", "yurut", "coz", "baslat", "bitir", "raporla", "paylas",
]
ADJS_POS = ["iyi", "guzel"]
ADJS_NEG = ["kotu", "cirkin"]


def tok(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")


def add_row(rows, idx, text, op, args):
    rows.append(
        {
            "id": f"g{idx:04d}",
            "text": text,
            "gold": [{"op": op, "args": [str(a) for a in args]}],
        }
    )


def build_rows(per_opcode: int = 30):
    rows = []
    idx = 1
    for i in range(per_opcode):
        a = SUBJECTS[i % len(SUBJECTS)]
        b = OBJECTS[i % len(OBJECTS)]
        add_row(rows, idx, f"{a} bir {b}dir", "ISA", [tok(a), tok(b)])
        idx += 1

    for i in range(per_opcode):
        a = SUBJECTS[(i + 3) % len(SUBJECTS)]
        b = OBJECTS[(i + 7) % len(OBJECTS)]
        add_row(rows, idx, f"{a} {b}dir", "ATTR", [tok(a), "ozellik", tok(b)])
        idx += 1

    for i in range(per_opcode):
        eff = OBJECTS[(i + 5) % len(OBJECTS)]
        cause = SUBJECTS[(i + 9) % len(SUBJECTS)]
        add_row(rows, idx, f"{eff} cunku {cause}", "CAUSE", [tok(cause), tok(eff)])
        idx += 1

    for i in range(per_opcode):
        cond = VERBS[i % len(VERBS)]
        eff = VERBS[(i + 1) % len(VERBS)]
        add_row(rows, idx, f"{cond} olursa {eff} olur", "CAUSE", [tok(cond), tok(eff)])
        idx += 1

    for i in range(per_opcode):
        s = SUBJECTS[(i + 1) % len(SUBJECTS)]
        add_row(rows, idx, f"{s} proje istiyor", "WANT", [tok(s), "hedef"])
        idx += 1

    for i in range(per_opcode):
        s = SUBJECTS[(i + 2) % len(SUBJECTS)]
        add_row(rows, idx, f"{s} plana inaniyor", "BELIEVE", [tok(s), "oneri", "0.7"])
        idx += 1

    for i in range(per_opcode):
        s = SUBJECTS[(i + 4) % len(SUBJECTS)]
        add_row(rows, idx, f"{s} cozum biliyor", "KNOW", [tok(s), "bilgi"])
        idx += 1

    for i in range(per_opcode):
        target = OBJECTS[(i + 6) % len(OBJECTS)]
        action = VERBS[(i + 2) % len(VERBS)]
        add_row(rows, idx, f"{target} icin {action}", "GOAL", [tok(action), tok(target), "medium"])
        idx += 1

    for i in range(per_opcode):
        a = VERBS[(i + 7) % len(VERBS)]
        b = VERBS[(i + 8) % len(VERBS)]
        add_row(rows, idx, f"once {a} sonra {b}", "BEFORE", [tok(a), tok(b)])
        idx += 1

    for i in range(per_opcode):
        a = OBJECTS[(i + 2) % len(OBJECTS)]
        b = OBJECTS[(i + 4) % len(OBJECTS)]
        add_row(rows, idx, f"{a} ama {b}", "OPPOSE", [tok(a), tok(b)])
        idx += 1

    for i in range(per_opcode):
        a = VERBS[(i + 10) % len(VERBS)]
        b = VERBS[(i + 11) % len(VERBS)]
        add_row(rows, idx, f"{a} engeller {b}", "PREVENT", [tok(a), tok(b)])
        idx += 1

    for i in range(per_opcode):
        a = OBJECTS[(i + 8) % len(OBJECTS)]
        adj = ADJS_POS[i % len(ADJS_POS)] if i % 2 == 0 else ADJS_NEG[i % len(ADJS_NEG)]
        score = "1.0" if adj in ADJS_POS else "-1.0"
        add_row(rows, idx, f"{a} {adj}", "EVAL", [tok(a), score])
        idx += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate Gold Set v1 dataset.")
    parser.add_argument("--per-opcode", type=int, default=30, help="Rows per opcode pattern block.")
    parser.add_argument("--output", default="data/gold/gold_set_v1.jsonl")
    args = parser.parse_args()

    rows = build_rows(per_opcode=max(1, int(args.per_opcode)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Gold set generated: {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
