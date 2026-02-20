#!/usr/bin/env python
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.listener import get_shadow_listener_status, start_shadow_listener
from core.vscode_chat_bridge import get_vscode_chat_bridge_status, start_vscode_chat_bridge


def main():
    poll = float(os.getenv("COGNITIVE_BRIDGE_POLL_SEC", "1.5"))
    start_shadow_listener(poll_interval=poll)
    start_vscode_chat_bridge(poll_interval=poll)
    print("[bridge-daemon] shadow listener + vscode bridge started")
    try:
        while True:
            s = get_shadow_listener_status()
            b = get_vscode_chat_bridge_status()
            print(
                f"[bridge-daemon] shadow_running={bool(s.get('running'))} "
                f"bridge_running={bool(b.get('running'))} "
                f"watched_files={int(b.get('watched_files', 0))} "
                f"forwarded_user={int(b.get('forwarded_user', 0))}"
            )
            time.sleep(max(1.0, poll * 4.0))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

