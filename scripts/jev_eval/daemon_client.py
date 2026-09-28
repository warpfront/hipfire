"""Minimal JSONL driver for target/release/daemon (decide gates + evals)."""
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT = 600.0  # seconds; first-use kernel JIT can take minutes
_EOF = object()


class DaemonTimeout(RuntimeError):
    pass


class Daemon:
    def __init__(self, binary=None, stderr=None, env=None, timeout=DEFAULT_TIMEOUT):
        """`env`: extra environment variables for the daemon process.
        `timeout`: default deadline (s) for each `recv_until`."""
        self.timeout = timeout
        self.p = subprocess.Popen(
            [str(binary or REPO / "target/release/daemon")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            env={**os.environ, **(env or {})}, text=True, bufsize=1)
        self.attempt = 0
        # A reader thread feeds stdout lines into a queue so recv_until can
        # wait with a deadline (select() on a buffered text pipe can miss
        # lines already sitting in the Python-side buffer).
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self):
        try:
            for line in self.p.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(_EOF)

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def recv_until(self, types, timeout=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DaemonTimeout(f"no {sorted(types)} reply within deadline")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise DaemonTimeout(f"no {sorted(types)} reply within deadline") from None
            if line is _EOF:
                self._lines.put(_EOF)  # keep EOF visible to later calls
                raise RuntimeError("daemon exited")
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v.get("type") in types:
                return v

    def load(self, model, max_seq=8192, params=None):
        """`params`: extra load params merged over `max_seq` (e.g. the cask*
        keys `hipfire serve` sends)."""
        self.send({"type": "load", "model": str(model),
                   "params": {"max_seq": max_seq, **(params or {})}})
        v = self.recv_until({"loaded", "error"})
        if v["type"] != "loaded":
            raise RuntimeError(f"load failed: {v}")
        return v

    def reset(self):
        self.attempt += 1
        self.send({"type": "reset", "attempt_id": self.attempt})
        v = self.recv_until({"reset", "error"})
        if v.get("type") != "reset" or not v.get("rolled_back"):
            raise RuntimeError(f"reset failed: {v}")

    def decide(self, state, questions, **extra):
        self.send({"type": "decide", "id": "g", "state": state, "questions": questions, **extra})
        # "error" covers uncorrelated daemon envelopes so a protocol slip fails
        # loudly instead of waiting for "decided" until the deadline.
        return self.recv_until({"decided", "error"})

    def generate_greedy(self, prompt, max_tokens=32):
        # Field names match the daemon "generate" arm (prompt, max_tokens,
        # temperature, thinking_enabled, mandatory attempt_id).
        self.attempt += 1
        self.send({"type": "generate", "id": f"gen{self.attempt}", "attempt_id": self.attempt,
                   "prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens,
                   "thinking_enabled": False})
        text = []
        while True:
            v = self.recv_until({"token", "done", "error"})
            if v["type"] == "token":
                text.append(v.get("text", ""))
            elif v["type"] == "done":
                return "".join(text)
            else:
                raise RuntimeError(f"generate failed: {v}")

    def close(self, timeout=120.0):
        """Unload (accepting `unloaded`, an `error` reply or a timeout), then
        always terminate; kill if the process does not exit."""
        try:
            if self.p.poll() is None:
                self.send({"type": "unload"})
                self.recv_until({"unloaded", "error"}, timeout=timeout)
        except Exception:  # noqa: BLE001 - shutdown must always proceed
            pass
        finally:
            self.p.terminate()
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
