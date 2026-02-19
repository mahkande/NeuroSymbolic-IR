#!/usr/bin/env python
import argparse

from core.file_listener import run_file_listener


def main():
    parser = argparse.ArgumentParser(description="Run watchdog listener for conversation.txt")
    parser.add_argument("--conversation", default="conversation.txt")
    parser.add_argument("--processed", default="memory/processed_logs.txt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--debounce", type=float, default=0.4)
    args = parser.parse_args()

    run_file_listener(
        conversation_path=args.conversation,
        processed_logs_path=args.processed,
        model_name=args.model,
        api_key=args.api_key,
        debounce_sec=args.debounce,
    )


if __name__ == "__main__":
    main()
