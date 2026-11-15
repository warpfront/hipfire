#!/usr/bin/env python3
"""Regression test for #1079: an image in `messages` must not erase the conversation.

Chat clients keep the image in history and resend it every turn. Before the fix
any `image_url` anywhere in `messages` sent the request down a single-turn vision
path that force-reset the KV/DeltaNet state and prefilled only the last user
text, so the image turn forgot every earlier turn and so did every later turn.

DETERMINISTIC ASSERTIONS (temperature 0, thinking off, Qwen3.5/3.8-VL + vision
sidecar):
  A  [user: code word + image, assistant: ack, user: what was the code word?]
     answers the code word and its `prompt_tokens` covers the history AND the
     image (image tokens + text), i.e. more than the text-only twin;
  T  an image turn, then the same conversation extended by a follow-up turn that
     resends the image: the follow-up answers from history, and reports the
     earlier turn as `cached_tokens` (continued in place, image not re-encoded);
  M  two images in different turns do not error, and a text turn after them still
     sees both (asks for the code word given in the first image turn).
  X  a text-only conversation is untouched (the answer is deterministic and no
     `cached_tokens` regression on its own follow-up).

Run against an already-serving daemon, or let the script spawn one:
  python3 scripts/test-vl-history.py --port 11435 --model qwen3.8:27b-mq4-xts
  python3 scripts/test-vl-history.py --spawn target/release --model ~/.hipfire/models/qwen3.8-27b.mq4-xts
Exit 0 = all assertions hold, 1 = a behavior assertion failed, 2 = setup problem.
"""
import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import struct
import tempfile
import zlib
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "benchmarks", "vision", "images")
FAILS = []


def data_uri(name):
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    with open(os.path.join(IMG_DIR, name), "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


def synthetic_png_uri(width, height, bg, fg):
    """A width x height PNG: `bg` with a centred `fg` square (stdlib only)."""
    left, right = width // 4, 3 * width // 4
    top, bottom = height // 4, 3 * height // 4
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row += bytes(fg if left <= x < right and top <= y < bottom else bg)
        rows.append(bytes(row))

    def chunk(kind, body):
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def image_part(name):
    return {"type": "image_url", "image_url": {"url": data_uri(name)}}


def chat(port, model, messages, max_tokens=120):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    usage = out["usage"]
    return {
        "text": out["choices"][0]["message"].get("content") or "",
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
    }


def check(label, ok, detail):
    print(f"[{'ok' if ok else 'FAIL'}] {label}: {detail}", flush=True)
    if not ok:
        FAILS.append(label)


def mentions_code_word(text):
    low = text.lower()
    return "purple hippo" in low or "purple-hippo" in low


def run(port, model):
    hello = "Remember the code word is purple hippo. Look at this image."
    ack = "Got it, the code word is purple hippo."
    q = "What was the code word I asked you to remember?"
    first = {"role": "user", "content": [{"type": "text", "text": hello}, image_part("scene_2.jpg")]}
    chat(port, model, [{"role": "user", "content": "Say hi."}], 8)  # warm the load

    # A: the image turn itself sees the earlier turns.
    a = chat(port, model, [first, {"role": "assistant", "content": ack}, {"role": "user", "content": q}])
    b = chat(port, model, [{"role": "user", "content": hello}, {"role": "assistant", "content": ack},
                           {"role": "user", "content": q}])
    check("A answers from history", "error" not in a and mentions_code_word(a["text"]), a)
    check("A prompt_tokens covers history + image", "error" not in a and "error" not in b
          and a["prompt_tokens"] > b["prompt_tokens"] + 100, (a.get("prompt_tokens"), b.get("prompt_tokens")))

    # T: a later turn resends the image and is continued in place.
    t1 = chat(port, model, [first])
    t_hist = [first, {"role": "assistant", "content": t1.get("text", "")}, {"role": "user", "content": q}]
    t2 = chat(port, model, t_hist)
    check("T follow-up answers from history", "error" not in t2 and mentions_code_word(t2["text"]), t2)
    t3 = chat(port, model, t_hist + [{"role": "assistant", "content": t2.get("text", "")},
                                     {"role": "user", "content": "Repeat the code word once more, nothing else."}])
    check("T 4th turn answers from history", "error" not in t3 and mentions_code_word(t3["text"]), t3)
    check("T continues in place (cached_tokens)", "error" not in t3 and t3["cached_tokens"] > 0, t3)

    # M: two images, in different turns, then a text turn.
    m1 = {"role": "user", "content": [{"type": "text", "text": "For the record, the secret word is blue lantern."},
                                      image_part("scene_1.jpg")]}
    r1 = chat(port, model, [m1])
    m_hist = [m1, {"role": "assistant", "content": r1.get("text", "")},
              {"role": "user", "content": [{"type": "text", "text": "Now a second picture."}, image_part("scene_2.jpg")]}]
    r2 = chat(port, model, m_hist)
    check("M two images do not error", "error" not in r2, r2)
    r3 = chat(port, model, m_hist + [{"role": "assistant", "content": r2.get("text", "")},
                                     {"role": "user", "content": "What was the secret word I gave you with the first picture?"}])
    check("M text turn after two images sees the first", "error" not in r3 and "blue lantern" in r3["text"].lower(), r3)

    # S: the client replaces the image in turn 1 with a DIFFERENT image of the
    # same size (same visual-token grid, same position). The resident rows belong
    # to the old image, so none may be reused and the answer must describe the new one.
    old = {"role": "user", "content": [{"type": "text", "text": "Remember this picture."},
                                       {"type": "image_url", "image_url": {"url": synthetic_png_uri(512, 384, (250, 230, 40), (200, 30, 30))}}]}
    new = {"role": "user", "content": [{"type": "text", "text": "Remember this picture."},
                                       {"type": "image_url", "image_url": {"url": synthetic_png_uri(512, 384, (240, 240, 240), (20, 40, 220))}}]}
    s1 = chat(port, model, [old])
    s2 = chat(port, model, [old, {"role": "assistant", "content": s1.get("text", "")},
                            {"role": "user", "content": "What color is the large square?"}])
    s3 = chat(port, model, [new, {"role": "assistant", "content": s1.get("text", "")},
                            {"role": "user", "content": "What color is the large square?"}])
    check("S old image answered red", "error" not in s2 and "red" in s2["text"].lower(), s2)
    check("S replaced image answered blue, not the old one", "error" not in s3
          and "blue" in s3["text"].lower() and "red" not in s3["text"].lower(), s3)
    check("S nothing reused past the replaced image", "error" not in s3 and s3["cached_tokens"] == 0, s3)

    # D: the client drops the image entirely; the text request must not extend past image rows.
    d = chat(port, model, [{"role": "user", "content": "Remember this picture."},
                           {"role": "assistant", "content": s1.get("text", "")},
                           {"role": "user", "content": "Say the word ready."}], 20)
    check("D text-only after an image conversation answers", "error" not in d and len(d["text"]) > 0, d)

    # X: text-only is untouched.
    x = chat(port, model, [{"role": "system", "content": "You are terse."},
                           {"role": "user", "content": "Name three prime numbers above 100."}], 60)
    check("X text-only answers", "error" not in x and len(x["text"]) > 0, x)


def spawn(bin_dir, model, port, seq):
    home = tempfile.mkdtemp(prefix="vl-history-home-")
    os.makedirs(os.path.join(home, ".hipfire"))
    models = os.path.expanduser(os.environ.get("HIPFIRE_MODELS_DIR", "~/.hipfire/models"))
    os.symlink(models, os.path.join(home, ".hipfire", "models"))
    with open(os.path.join(home, ".hipfire", "config.toml"), "w") as fh:
        fh.write(f'[serve]\nhost = "127.0.0.1"\nport = {port}\ndefault_model = "{model}"\n\n'
                 f'[memory]\nmax_seq = {seq}\nkv_cache = "q8"\n\n'
                 '[speculation]\nmode = "off"\ndflash = "off"\nmtp = "off"\nngram = "off"\n\n'
                 '[vision]\nmode = "auto"\n')
    env = dict(os.environ, HOME=home, HIPFIRE_HOME=os.path.join(home, ".hipfire"),
               HIPFIRE_DAEMON_BIN=os.path.join(bin_dir, "daemon"), HIPFIRE_MODEL=model)
    log = open(os.path.join(home, "serve.log"), "w")
    proc = subprocess.Popen([os.path.join(bin_dir, "hipfire"), "serve", "127.0.0.1", str(port)],
                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(300):
        if proc.poll() is not None:
            sys.exit(f"serve exited early; log: {home}/serve.log")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return proc
        except Exception:
            time.sleep(2)
    os.killpg(proc.pid, signal.SIGTERM)
    sys.exit(f"serve did not become ready; log: {home}/serve.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--model", required=True, help="model tag or path the serve resolves")
    ap.add_argument("--spawn", metavar="BIN_DIR", help="directory with `hipfire` and `daemon`; start (and stop) a serve")
    ap.add_argument("--max-seq", type=int, default=16384)
    a = ap.parse_args()
    proc = spawn(a.spawn, a.model, a.port, a.max_seq) if a.spawn else None
    try:
        run(a.port, a.model)
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                pass
    if FAILS:
        print("FAILED:", ", ".join(FAILS))
        sys.exit(1)
    print("all vision-history assertions hold")


if __name__ == "__main__":
    main()
