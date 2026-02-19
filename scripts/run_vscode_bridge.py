#!/usr/bin/env python
import argparse
import time

from core.vscode_chat_bridge import get_vscode_chat_bridge_status, start_vscode_chat_bridge, stop_vscode_chat_bridge


def main():
    parser = argparse.ArgumentParser(description="Run VSCode/Cursor chat bridge")
    parser.add_argument("--poll", type=float, default=1.5)
    parser.add_argument("--lookback-days", type=int, default=30)
    args = parser.parse_args()

    start_vscode_chat_bridge(poll_interval=args.poll, lookback_days=args.lookback_days)
    try:
        while True:
            st = get_vscode_chat_bridge_status()
            print(
                f"[vscode-bridge] running={st.get('running')} watched={st.get('watched_files')} "
                f"user={st.get('forwarded_user')} assistant={st.get('forwarded_assistant')}"
            )
            time.sleep(max(1.0, args.poll * 2))
    except KeyboardInterrupt:
        pass
    finally:
        stop_vscode_chat_bridge()


if __name__ == "__main__":
    main()
