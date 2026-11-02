#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Fake hipfire daemon enforcing the real generate wire rules (CPU test double).

Rules mirrored from crates/hipfire-daemon/src/main.rs and
crates/hipfire-engine/src/terminal.rs:
  * generate without a numeric nonzero attempt_id -> {"type":"error"} (validation)
  * commit_ready is emitted after the tokens; a commit that arrives BEFORE
    commit_ready is ignored
  * no matching commit within the commit timeout -> {"type":"error"} abort,
    never a done
  * done is emitted only after a matching {"type":"commit","id","attempt_id"}

Environment:
  FAKE_COMMIT_TIMEOUT  seconds to wait for the commit (default 2.0)
  FAKE_READY_DELAY     seconds between the last token and commit_ready (default 0)
  FAKE_HANG            "1": load works, every generate is swallowed (deadline test)
  FAKE_PIDFILE         if set, the daemon writes its pid there on start
  FAKE_FAULT           eof_before_done | hang_after_tokens | error_after_tokens |
                       error_null_id: once request FAKE_FAULT_ID (default c0_t1) has
                       emitted its tokens: exit 0 (EOF), go silent, emit an error with
                       that id, or emit an error with id null
  FAKE_EXIT_AFTER_DONE N: exit FAKE_EXIT_CODE (default 1) after the Nth done
  FAKE_EXIT_ON_UNLOAD  "1": reply `unloaded`, then exit FAKE_EXIT_CODE
  FAKE_EOF_ON_UNLOAD   "1": exit 0 on unload without replying `unloaded`
  FAKE_ECHO_ENV        comma-separated env var names echoed in the `loaded` event
"""
import json
import os
import queue
import sys
import threading
import time

COMMIT_TIMEOUT = float(os.environ.get("FAKE_COMMIT_TIMEOUT", "2.0"))
READY_DELAY = float(os.environ.get("FAKE_READY_DELAY", "0"))
HANG = os.environ.get("FAKE_HANG") == "1"
FAULT = os.environ.get("FAKE_FAULT", "")
FAULT_ID = os.environ.get("FAKE_FAULT_ID", "c0_t1")
EXIT_CODE = int(os.environ.get("FAKE_EXIT_CODE", "1"))
EXIT_AFTER_DONE = int(os.environ.get("FAKE_EXIT_AFTER_DONE", "0"))
EXIT_ON_UNLOAD = os.environ.get("FAKE_EXIT_ON_UNLOAD") == "1"
EOF_ON_UNLOAD = os.environ.get("FAKE_EOF_ON_UNLOAD") == "1"

TOOL_CALL = '<tool_call>{"name":"read_file","arguments":{"path":"main.c"}}</tool_call>'
INLINE = "The file defines main and returns zero."


def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def valid_attempt(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def main():
    if os.environ.get("FAKE_PIDFILE"):
        with open(os.environ["FAKE_PIDFILE"], "w") as f:
            f.write(str(os.getpid()))
    lines = queue.Queue()
    dones = 0

    def reader():
        for raw in sys.stdin:
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    sys.stderr.write("fake-daemon: started\n")
    sys.stderr.flush()

    while True:
        raw = lines.get()
        if raw is None:
            return 0
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        typ = msg.get("type")
        if typ == "load":
            echo = [k for k in os.environ.get("FAKE_ECHO_ENV", "").split(",") if k]
            emit({"type": "loaded", "arch": "fake", "model": msg.get("model"),
                  "env": {k: os.environ.get(k) for k in echo}})
        elif typ == "unload":
            if EOF_ON_UNLOAD:
                return 0  # exits without `unloaded`
            emit({"type": "unloaded"})
            if EXIT_ON_UNLOAD:
                return EXIT_CODE
        elif typ == "commit":
            continue  # no generate awaiting a commit: ignored
        elif typ == "generate":
            rid, attempt = msg.get("id"), msg.get("attempt_id")
            if HANG:
                continue
            if not valid_attempt(attempt):
                emit({"type": "error", "id": rid, "attempt_id": 0,
                      "class": "validation",
                      "message": "generate attempt_id must be a nonzero number"})
                continue
            text = INLINE if str(rid).endswith("_t2") else TOOL_CALL
            emit({"type": "token", "id": rid, "text": text})
            if FAULT and rid == FAULT_ID:
                if FAULT == "eof_before_done":
                    return 0  # valid tool-call tokens, then stdout EOF
                if FAULT == "hang_after_tokens":
                    continue  # valid tokens, then silence: the driver's deadline fires
                if FAULT == "error_after_tokens":
                    emit({"type": "error", "id": rid, "attempt_id": attempt,
                          "class": "fault", "message": "fake fault after tokens"})
                    continue
                if FAULT == "error_null_id":
                    emit({"type": "error", "id": None, "class": "fault",
                          "message": "fake fault with no id"})
                    continue
            if READY_DELAY:
                time.sleep(READY_DELAY)
            while True:  # a commit sent before commit_ready is ignored
                try:
                    early = lines.get_nowait()
                except queue.Empty:
                    break
                if early is None:
                    return 0
            emit({"type": "commit_ready", "id": rid, "attempt_id": attempt,
                  "finish_reason": "stop"})
            deadline = time.monotonic() + COMMIT_TIMEOUT
            committed = False
            while True:
                try:
                    nxt = lines.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if nxt is None:
                    return 0
                try:
                    reply = json.loads(nxt)
                except ValueError:
                    continue
                if (reply.get("type") == "commit" and reply.get("id") == rid
                        and reply.get("attempt_id") == attempt):
                    committed = True
                    break
            if committed:
                emit({"type": "done", "id": rid, "attempt_id": attempt,
                      "tokens": 1, "finish_reason": "stop"})
                dones += 1
                if EXIT_AFTER_DONE and dones == EXIT_AFTER_DONE:
                    return EXIT_CODE
            else:
                emit({"type": "error", "id": rid, "attempt_id": attempt,
                      "class": "abort",
                      "message": "client commit timeout; request aborted"})


if __name__ == "__main__":
    sys.exit(main())
