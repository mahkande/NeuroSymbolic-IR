#!/usr/bin/env python
import argparse
from datetime import datetime, timezone
from pathlib import Path


def read_text(args):
    if args.text:
        return args.text
    if args.clipboard:
        import pyperclip  # optional

        return (pyperclip.paste() or "").strip()
    data = "\n".join([line.rstrip("\n") for line in __import__("sys").stdin.readlines()]).strip()
    return data


def main():
    parser = argparse.ArgumentParser(description="Append chat content to conversation.txt")
    parser.add_argument("--text", type=str, default="", help="Text to append")
    parser.add_argument("--clipboard", action="store_true", help="Read text from clipboard via pyperclip")
    parser.add_argument("--source", type=str, default="cursor", help="Source label")
    parser.add_argument("--file", type=str, default="conversation.txt", help="Output conversation file")
    args = parser.parse_args()

    text = read_text(args)
    if not text:
        raise SystemExit("No text to append.")

    p = Path(args.file).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    block = f"[{stamp}] source={args.source}\n{text}\n\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(block)
    print(f"Appended to {p}")


if __name__ == "__main__":
    main()
