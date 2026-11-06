#!/usr/bin/env python3
"""Served `generate` identity + decode tok/s harness (retained PM4 vs HIP graph).

Drives the product daemon over its stdio protocol exactly like `serve` does,
INCLUDING the terminal handshake: the daemon stages the turn, emits
`commit_ready`, and waits (30 s) for a matching `commit` before it sends `done`.
A client that never acks makes every turn abort, roll back, and invalidate the
request-local replay observation window, so no HIPFIRE_REPLAY_ROUTE_PROOF marker
is printed even when the retained route replayed every token.

  tools/redline/serve_identity_ab.py ident --daemon D --model M --out DIR [--tokens 512]
  tools/redline/serve_identity_ab.py ab    --daemon D --model M --out DIR [--rounds 2] [--tokens 512]

Arms (each a fresh daemon + model load):
  on    HIPFIRE_REPLAY_BACKEND=redline                           (retained PM4 + scratch rings)
  hip   HIPFIRE_REPLAY_BACKEND=hip                               (HIP graph)
  kill  HIPFIRE_REPLAY_BACKEND=redline HIPFIRE_PMSCRATCH_PM4_SCRATCH=0 (route falls back to HIP)
  dflt  no HIPFIRE_REPLAY_BACKEND at all (the product default decision)
"""
import argparse, hashlib, json, os, re, select, signal, subprocess, sys, time

STORY = "Write a long story about a lighthouse keeper who discovers a message in a bottle."
CODE = "Write a Rust function that parses a CSV line with quoted fields, and explain each step."
LONG = ("The following is a long log of unrelated notes. " + " ".join(
    f"Note {i}: the archive room smells of cedar, entry {i*7%101} was filed under section {i%13}." for i in range(190))
    + "\n\nQuestion: continue the log with at least forty more notes in exactly the same style, numbered consecutively.")

BASE_ENV = {"HIPFIRE_EMIT_TOKEN_IDS": "1", "HIPFIRE_REPLAY_ROUTE_PROOF_LOG": "1",
            "HIPFIRE_AR_GRAPH": "1", "HIPFIRE_GRAPH": "1", "HIPFIRE_KV_MODE": "q8"}
ARMS = {
    "on": {"HIPFIRE_REPLAY_BACKEND": "redline", "HIPFIRE_REPLAY_TRANSPORT": "pm4"},
    "hip": {"HIPFIRE_REPLAY_BACKEND": "hip"},
    "kill": {"HIPFIRE_REPLAY_BACKEND": "redline", "HIPFIRE_REPLAY_TRANSPORT": "pm4", "HIPFIRE_PMSCRATCH_PM4_SCRATCH": "0"},
    "dflt": {},
}


class Daemon:
    def __init__(self, binary, arm, log, timeout):
        env = dict(os.environ)
        for k in ("HIPFIRE_REPLAY_BACKEND", "HIPFIRE_REPLAY_TRANSPORT", "HIPFIRE_PMSCRATCH_PM4_SCRATCH",
                  "HIPFIRE_REPLAY_MANUAL_CAPTURE"):
            env.pop(k, None)
        env.update(BASE_ENV)
        env.update(ARMS[arm])
        self.timeout = timeout
        self.log = log
        self.p = subprocess.Popen([binary], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(log, "w"), text=True, bufsize=1, start_new_session=True)

    def send(self, m):
        self.p.stdin.write(json.dumps(m, separators=(",", ":")) + "\n")
        self.p.stdin.flush()

    def recv(self):
        r, _, _ = select.select([self.p.stdout], [], [], self.timeout)
        if not r:
            raise TimeoutError("daemon silent")
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("daemon exited")
        try:
            return json.loads(line)
        except Exception:
            return None

    def load(self, model, max_seq):
        self.send({"type": "load", "model": model, "params": {"max_seq": max_seq, "kv_mode": "q8", "dflash_mode": "off",
                   "mtp_mode": "off", "dspark_mode": "off", "tp": 1, "pp": 1}})
        while True:
            j = self.recv()
            if j and j.get("type") == "error":
                raise RuntimeError(j.get("message"))
            if j and j.get("type") == "loaded":
                return j

    def generate(self, rid, prompt, n):
        self.send({"type": "generate", "id": rid, "attempt_id": 1, "prompt": prompt, "temperature": 0.0, "top_p": 1.0,
                   "repeat_penalty": 1.0, "max_tokens": n, "max_think_tokens": 1, "assistant_prefix": "closed_think"})
        ids, stamps, done, acked = [], [], None, False
        while True:
            j = self.recv()
            if not j:
                continue
            t = j.get("type")
            if t == "committed":
                ids.append(j["tok_id"])
                stamps.append(time.monotonic())
            elif t == "commit_ready":
                self.send({"type": "commit", "id": j.get("id", rid), "attempt_id": j.get("attempt_id", 1)})
                acked = True
            elif t == "done":
                done = j
                break
            elif t == "error":
                raise RuntimeError(j.get("message"))
        wall_tps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        return {"ids": ids, "done": done, "acked": acked, "wall_tok_s": wall_tps}

    def close(self):
        try:
            self.send({"type": "unload"})
        except Exception:
            pass
        try:
            os.killpg(self.p.pid, signal.SIGTERM)
            self.p.wait(timeout=30)
        except Exception:
            os.killpg(self.p.pid, signal.SIGKILL)
            self.p.wait(timeout=30)


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def proofs(log):
    text = open(log).read()
    marks = {}
    for m in re.finditer(r"HIPFIRE_REPLAY_ROUTE_PROOF transport=(\S+) position=(\d+) request_id=(\S+) replays=(\d+)", text):
        marks[m.group(3)] = {"transport": m.group(1), "position": int(m.group(2)), "replays": int(m.group(4))}
    return {
        "marks": marks,
        "scratch_ring_lines": len(re.findall(r"PM4 scratch ring", text)),
        "fallbacks": sorted(set(re.findall(r"falling back to HIP: (.*)", text))),
        "retained_default_line": len(re.findall(r"enabling fail-closed retained default", text)),
        "redline_lines": len(re.findall(r"^\[redline\]", text, re.M)),
        "queue_timeouts": text.count("QUEUE_TIMEOUT"),
    }


def ident(args):
    prompts = [("story", STORY), ("code", CODE), ("long", LONG), ("story2", STORY)]
    res, ok = {}, True
    for arm in args.arms.split(","):
        log = f"{args.out}/ident-{arm}.log"
        d = Daemon(args.daemon, arm, log, args.timeout)
        rows = {}
        try:
            d.load(args.model, args.max_seq)
            for name, prompt in prompts:
                r = d.generate(f"{arm}-{name}", prompt, args.tokens)
                rows[name] = r
        finally:
            d.close()
        res[arm] = {"rows": rows, "proof": proofs(log)}
    report = {"prompts": {}, "arms": {}, "pass": True, "failures": []}
    ref = args.arms.split(",")[0]
    for name, _ in prompts:
        shas = {arm: sha(res[arm]["rows"][name]["ids"]) for arm in res}
        ntok = {arm: len(res[arm]["rows"][name]["ids"]) for arm in res}
        same = len(set(shas.values())) == 1 and all(n > 0 for n in ntok.values())
        replays = {arm: res[arm]["proof"]["marks"].get(f"{arm}-{name}", {}).get("replays") for arm in res}
        report["prompts"][name] = {"tokens": ntok, "sha": shas, "identical": same, "replays": replays,
                                   "finish": {arm: res[arm]["rows"][name]["done"].get("finish_reason") for arm in res},
                                   "prompt_tokens": {arm: res[arm]["rows"][name]["done"].get("prefill_tokens") for arm in res},
                                   "acked": {arm: res[arm]["rows"][name]["acked"] for arm in res}}
        if not same:
            report["failures"].append(f"{name}: token ids differ")
    for arm in res:
        p = res[arm]["proof"]
        report["arms"][arm] = {k: v for k, v in p.items() if k != "marks"}
        report["arms"][arm]["marks"] = p["marks"]
        if arm in ("on", "dflt"):
            for name, _ in prompts:
                m = p["marks"].get(f"{arm}-{name}")
                if not m or m["replays"] <= 0:
                    report["failures"].append(f"{arm}/{name}: no ROUTE_PROOF replays>0")
            if p["scratch_ring_lines"] == 0:
                report["failures"].append(f"{arm}: no PM4 scratch ring line")
        if arm == "dflt" and p["retained_default_line"] == 0:
            report["failures"].append("dflt: no 'enabling fail-closed retained default' line (default did not select the route)")
        if arm != "dflt" and p["retained_default_line"] != 0:
            report["failures"].append(f"{arm}: unexpected automatic retained default")
        if arm == "kill":
            if p["marks"]:
                report["failures"].append("kill: unexpected ROUTE_PROOF markers")
            if not p["fallbacks"]:
                report["failures"].append("kill: no 'falling back to HIP' message")
        if arm == "hip":
            if p["marks"] or p["redline_lines"]:
                report["failures"].append("hip: retained route engaged")
    report["pass"] = not report["failures"]
    json.dump(report, open(f"{args.out}/ident-report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


def ab(args):
    arms = args.arms.split(",")
    order = []
    for r in range(args.rounds):
        order += arms + arms[::-1]
    out = []
    for i, arm in enumerate(order):
        log = f"{args.out}/ab-{i:02d}-{arm}.log"
        d = Daemon(args.daemon, arm, log, args.timeout)
        row = {"index": i, "arm": arm, "story": [], "long": []}
        try:
            d.load(args.model, args.max_seq)
            d.generate(f"ab{i}-warm", STORY, args.tokens)
            for k in range(args.reps):
                r = d.generate(f"ab{i}-story{k}", STORY, args.tokens)
                row["story"].append({"tokens": len(r["ids"]), "decode_tok_s": r["done"].get("decode_tok_s"),
                                     "wall_tok_s": round(r["wall_tok_s"], 3), "sha": sha(r["ids"]), "finish": r["done"].get("finish_reason")})
            for k in range(args.long_reps):
                r = d.generate(f"ab{i}-long{k}", LONG, args.tokens)
                row["long"].append({"tokens": len(r["ids"]), "prompt_tokens": r["done"].get("prefill_tokens"),
                                    "decode_tok_s": r["done"].get("decode_tok_s"), "wall_tok_s": round(r["wall_tok_s"], 3),
                                    "sha": sha(r["ids"]), "finish": r["done"].get("finish_reason")})
        finally:
            d.close()
        p = proofs(log)
        row["replays"] = {k: v["replays"] for k, v in p["marks"].items()}
        row["fallbacks"] = p["fallbacks"]
        row["scratch_ring_lines"] = p["scratch_ring_lines"]
        out.append(row)
        print(json.dumps({k: row[k] for k in ("index", "arm")} | {"story_dec": [x["decode_tok_s"] for x in row["story"]],
              "long_dec": [x["decode_tok_s"] for x in row["long"]], "n_replays": len(row["replays"])}), flush=True)
    json.dump(out, open(f"{args.out}/ab-report.json", "w"), indent=1)
    import statistics
    summary = {}
    for arm in arms:
        rows = [r for r in out if r["arm"] == arm]
        for kind in ("story", "long"):
            vals = [x["decode_tok_s"] for r in rows for x in r[kind] if x["decode_tok_s"]]
            walls = [x["wall_tok_s"] for r in rows for x in r[kind]]
            if vals:
                summary[f"{arm}/{kind}"] = {"decode_median": statistics.median(vals), "wall_median": statistics.median(walls), "n": len(vals)}
        summary[f"{arm}/replays_rows"] = sum(1 for r in rows if r["replays"])
    json.dump(summary, open(f"{args.out}/ab-summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["ident", "ab"])
ap.add_argument("--daemon", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--timeout", type=float, default=300)
ap.add_argument("--arms", default=None)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--long-reps", type=int, default=2)
a = ap.parse_args()
if a.arms is None:
    a.arms = "on,hip,kill" if a.mode == "ident" else "on,hip"
os.makedirs(a.out, exist_ok=True)
sys.exit(ident(a) if a.mode == "ident" else ab(a))
