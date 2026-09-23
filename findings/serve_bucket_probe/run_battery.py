#!/usr/bin/env python3
"""Deterministic native-serve prefill bucket probe (investigation only)."""
import hashlib, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/home/kaden/hipfire-strix-prefill")
OUT = ROOT / "findings/serve_bucket_probe"
PROMPTS = OUT / "prompts"
MODEL = os.environ.get("HIPFIRE_MODEL", str(Path.home() / ".hipfire/models/qwen3.6-35b-a3b.mq4r"))
PORT = int(os.environ.get("PORT", "11571"))
HOME = str(OUT / "harness_home")
SERVE_LOG = str(OUT / "serve.log")
CLI = os.environ.get("HIPFIRE_CLI_BIN", str(ROOT / "target/release/hipfire"))
DAEMON = os.environ.get("HIPFIRE_DAEMON_BIN", str(ROOT / "target/release/examples/daemon"))
HIP_DEV = os.environ.get("HIP_VISIBLE_DEVICES", "1")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
SEED = 42

# Import harness helpers without running main
sys.path.insert(0, str(ROOT / "scripts"))
import serve_harness as sh

RESULTS = []


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_text(s: str):
    return hashlib.md5(s.encode()).hexdigest()


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return json.load(r)
    except Exception as e:
        return {"error": str(e)}


def send(messages, label, prompt_text=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "seed": SEED,
        "reasoning_effort": "none",
    }
    t0 = time.time()
    ttft = None
    ans = []
    think = []
    usage = {}
    timings = {}
    finish = None
    completion_id = None
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                ck = json.loads(p)
            except Exception:
                continue
            if isinstance(ck.get("id"), str):
                completion_id = ck["id"]
            if ck.get("usage"):
                usage = ck["usage"]
            if ck.get("timings"):
                timings = ck["timings"]
            ch = (ck.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            d = ch.get("delta") or {}
            if isinstance(d.get("reasoning_content"), str) and d["reasoning_content"]:
                if ttft is None:
                    ttft = time.time() - t0
                think.append(d["reasoning_content"])
            if isinstance(d.get("content"), str) and d["content"]:
                if ttft is None:
                    ttft = time.time() - t0
                ans.append(d["content"])
    wall = time.time() - t0
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    row = {
        "label": label,
        "request_id": completion_id,
        "prompt_md5": md5_text(prompt_text) if prompt_text is not None else None,
        "request_body_md5": md5_text(json.dumps(body, sort_keys=True)),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prefill_ms": timings.get("prefill_ms"),
        "prefill_tok_s": timings.get("prefill_tok_s"),
        "decode_tok_s": timings.get("decode_tok_s"),
        "timings": timings,
        "usage": usage,
        "finish": finish,
        "ttft_s": round(ttft or 0, 3),
        "wall_s": round(wall, 3),
        "ans_preview": ("".join(ans) or "".join(think))[:120],
        "classification": None,
    }
    # classify
    if cached and cached > 0 and cached >= max(0, prompt_tokens - 8):
        row["classification"] = "cache-hit-or-near-full-prefix"
    elif cached and cached > 0:
        row["classification"] = "partial-prefix-cache"
    elif "cold" in label or label.endswith("_1"):
        row["classification"] = "cold-or-first"
    else:
        row["classification"] = "warm-no-cache-or-uncached"
    RESULTS.append(row)
    print(json.dumps({k: row[k] for k in [
        "label","prompt_tokens","cached_tokens","prefill_ms","prefill_tok_s","wall_s","classification","ans_preview"
    ]}, ensure_ascii=False), flush=True)
    return row, "".join(ans) or "".join(think)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "commit": subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"], text=True).strip(),
        "model": MODEL,
        "model_md5": md5_file(MODEL),
        "cli": CLI,
        "cli_md5": md5_file(CLI) if os.path.isfile(CLI) else None,
        "daemon": DAEMON,
        "daemon_md5": md5_file(DAEMON) if os.path.isfile(DAEMON) else None,
        "HIP_VISIBLE_DEVICES": HIP_DEV,
        "port": PORT,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "started_unix": time.time(),
    }
    (OUT / "run_meta_start.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("META", json.dumps(meta, indent=2), flush=True)

    # Build cfg for harness spawn
    cfg = {
        "model": MODEL,
        "tag": "qwen3.6:35b-a3b",
        "kv": "fwht3",
        "mtp": "off",
        "kv_backend": "contiguous",
        "dflash": "off",
        "draft": None,
        "thinking_budget": "low",
        "thinking_cap_tokens": 512,
        "max_tokens": MAX_TOKENS,
        "sampling": {"temperature": 0.0, "reasoning_effort": "none"},
        "sampling_source": {"temperature": "explicit", "reasoning_effort": "explicit"},
        "mode": "battery",
        "port": PORT,
        "seed": SEED,
        "prompts_file": None,
        "replay_route_proof_log": False,
        "max_seq": 32768,
    }
    os.environ["HIP_VISIBLE_DEVICES"] = HIP_DEV
    os.environ["HIPFIRE_CLI_BIN"] = CLI
    os.environ["HIPFIRE_DAEMON_BIN"] = DAEMON
    # clear old log
    open(SERVE_LOG, "w").close()
    print("SPAWNING serve...", flush=True)
    off = sh.spawn_serve(cfg, HOME, SERVE_LOG)
    if off is None:
        print(open(SERVE_LOG).read()[-8000:], file=sys.stderr)
        sys.exit("serve failed to warm")
    h = health()
    print("HEALTH", h, flush=True)
    (OUT / "health.json").write_text(json.dumps(h, indent=2) + "\n")

    # Token-length buckets: refine prompts if needed? use existing files.
    buckets = [232, 1024, 4096, 8192]
    for b in buckets:
        text = (PROMPTS / f"prompt_{b}.txt").read_text()
        pmd5 = md5_text(text)
        msgs = [{"role": "user", "content": text}]
        # cold
        send(msgs, f"bucket_{b}_cold", text)
        # warm byte-identical
        send(msgs, f"bucket_{b}_warm", text)
        # tiny pause to make log ordering clear
        time.sleep(0.2)

    # growing-context multi-turn session (prefix cache exposure)
    base_bits = [
        "Turn1: Name three primary colors.",
        "Turn2: Now add the secondary colors made from those primaries.",
        "Turn3: Give one real-world example object for each color listed so far.",
        "Turn4: Compress the full color list into a single comma-separated line.",
        "Turn5: Repeat only the secondary colors from that line.",
    ]
    messages = []
    for i, t in enumerate(base_bits, 1):
        messages.append({"role": "user", "content": t})
        row, ans = send(messages, f"multiturn_t{i}", t)
        messages.append({"role": "assistant", "content": ans or "(empty)"})

    # snapshot serve log tail + full
    log_txt = Path(SERVE_LOG).read_text(errors="replace")
    (OUT / "serve_full.log").write_text(log_txt)
    # extract useful lines
    keep = []
    for line in log_txt.splitlines():
        if re.search(r"prefill|cache|cached|route|KV cache|prefix|tok/s|JIT|compile|warm|cold|request|prompt", line, re.I):
            keep.append(line)
    (OUT / "serve_filtered.log").write_text("\n".join(keep) + "\n")

    out_path = OUT / "results.json"
    payload = {"meta": meta, "health": h, "rows": RESULTS}
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    # table
    print("\n=== TABLE ===", flush=True)
    hdr = f"{label:28} {ptok:>6} {cached:>6} {pf_ms:>10} {pf_tok_s:>10} {wall_s:>8} class"
    print(hdr, flush=True)
    for r in RESULTS:
        print(f"{r[label]:28} {r[prompt_tokens]:>6} {r[cached_tokens]:>6} {str(r[prefill_ms]):>10} {str(r[prefill_tok_s]):>10} {r[wall_s]:>8} {r[classification]}", flush=True)

    # classification summary
    near_160 = [r for r in RESULTS if isinstance(r.get("prefill_tok_s"), (int, float)) and r["prefill_tok_s"] < 250]
    summary = {
        "n_rows": len(RESULTS),
        "near_or_below_250_tok_s": [
            {k: r[k] for k in ("label","prompt_tokens","cached_tokens","prefill_ms","prefill_tok_s","classification")}
            for r in near_160
        ],
        "min_prefill_tok_s": min((r["prefill_tok_s"] for r in RESULTS if isinstance(r.get("prefill_tok_s"), (int, float))), default=None),
        "max_prefill_tok_s": max((r["prefill_tok_s"] for r in RESULTS if isinstance(r.get("prefill_tok_s"), (int, float))), default=None),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)

    sh._kill_serve()
    print("DONE", out_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            sh._kill_serve()
        except Exception:
            pass
