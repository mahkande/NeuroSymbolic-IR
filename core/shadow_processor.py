import argparse
import signal
import time

from core.listener import get_shadow_listener_status, start_shadow_listener, stop_shadow_listener


def main():
    parser = argparse.ArgumentParser(description="Shadow Processor (file-based chat watcher)")
    parser.add_argument("--poll", type=float, default=1.5, help="Polling interval seconds")
    parser.add_argument("--model", type=str, default=None, help="Groq model name")
    parser.add_argument(
        "--watch",
        type=str,
        default="",
        help="Semicolon-separated file list. If empty, defaults include conversation.txt",
    )
    args = parser.parse_args()

    watch_files = [p.strip() for p in args.watch.split(";") if p.strip()] if args.watch else None
    start_shadow_listener(watch_files=watch_files, poll_interval=args.poll, model=args.model)

    running = True

    def _stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        status = get_shadow_listener_status()
        print(
            f"[shadow] running={status.get('running')} processed={status.get('processed_blocks')} "
            f"produced_ir={status.get('produced_ir')} rejected={status.get('rejected_ir')}"
        )
        time.sleep(max(1.0, args.poll * 2))

    stop_shadow_listener()


if __name__ == "__main__":
    main()
