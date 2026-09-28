"""Minimal JSONL driver for target/release/daemon (decide gates + evals)."""
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class Daemon:
    def __init__(self, binary=None, stderr=None, env=None):
        """`env`: extra environment variables for the daemon process."""
        self.p = subprocess.Popen(
            [str(binary or REPO / "target/release/daemon")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            env={**os.environ, **(env or {})}, text=True, bufsize=1)
        self.attempt = 0

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def recv_until(self, types):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("daemon exited")
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v.get("type") in types:
                return v

    def load(self, model, max_seq=8192):
        self.send({"type": "load", "model": str(model), "params": {"max_seq": max_seq}})
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
        # loudly instead of blocking forever waiting for "decided".
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

    def close(self):
        try:
            if self.p.poll() is None:
                self.send({"type": "unload"})
                self.recv_until({"unloaded"})
        finally:
            self.p.terminate()
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
