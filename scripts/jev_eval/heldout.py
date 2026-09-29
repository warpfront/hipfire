"""Held-out labelled rows for fitting decide calibration (spec §13.5).

  build   draw candidates, drop every one whose state is a reported row's,
          keep PER_SOURCE per source. Examples (with text) go to the work dir;
          the text-free manifest and the reported-state hash list go to
          bench/jev/heldout/.
  verify  re-check the committed hash list and manifest against the
          work-dir examples: no held-out state is a reported state.
  run     ask a running serve every held-out example, one question per
          request as the reported runs did; write text-free answer rows.

The external clones are read-only: jevbench's cache ROOT is redirected to the
work dir, and no bytecode is written next to modules imported from them.

Licence note (bench/jev/DATA-LICENSES.md, human decision): ag-news,
duplicates (QQP), yelp-stars, offensive (TweetEval), sentiment-it
(CardiffNLP Italian) and hellaswag are restricted sources. Their held-out
*examples* (text) are built here like any other source, but only ever land
in the gitignored work dir, same as every other source's. What this module
commits under bench/jev/heldout/ — manifest.json and
reported_state_sha256.txt — holds no dataset text and no per-row probs or
labels, only aggregate counts and one-way SHA-256 hashes, so it is
provenance metadata rather than derived data (the same rule
bench/jev/DATA-LICENSES.md already applies to probs/SOURCES.json). The
per-model answer rows `run` writes under bench/jev/<model>/heldout/ are
derived data (probs + gold) and follow probs/'s committed-iff-licence-clear
rule; see the .gitignore entry next to this file's probs/ counterpart.

`run` asks only licence-clear sources by default (licences.py, the single
choke point); build/verify still cover all 17 so the disjointness proof and
the committed manifest are unchanged.

noul keys: jev-bench noul rows carry keys ["true", "false"] (keys_for) and
calsets-built ones ["yes", "no"]. Both put the true answer at index 0, and
the answer rows never store keys: probs is [noul, 1 - noul] and gold is 0 for
a true statement either way (see test_heldout.NoulKeys)."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
import calsets  # noqa: E402  (scripts/jev_eval is sys.path[0])
import licences  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
CAL = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))
EVAL = Path(os.environ.get("JEV_EVAL_DIR", Path.home() / "repos/hipfire-jev-eval/bench/jev"))
WORK = Path(os.environ.get("JEV_CALIB_WORK", Path.home() / ".cache/hipfire-jev-calib"))
OUT = REPO / "bench/jev/heldout"
HELDOUT_SEED = 1  # jevbench.SEED is 0
SYNTH_SEED = 7  # generate.py's documented fresh seed; data/val.jsonl is seed 0
SYNTH_TICKETS = 400
DRAW = 400
PER_SOURCE = 300
REPORTED_MODELS = ("qwen3.5-4b", "qwen3.8-27b-mq4-xts")
# jev-bench task -> held-out split (spec §13.5 table).
JEVBENCH_SPLIT = {
    "banking77": "train", "massive-en": "validation", "massive-it": "validation",
    "clinc150": "validation", "ledgar": "validation", "ag-news": "train",
    "sms-spam": "train", "duplicates": "train", "doc-yesno": "train",
    "offensive": "validation", "yelp-stars": "train", "sentiment-it": "validation",
}
PUBLIC_TRAIN = ("openbookqa", "commonsense_qa")


def state_hash(state):
    """SHA-256 of the state text: strings as-is, objects as sorted compact JSON."""
    text = state if isinstance(state, str) else json.dumps(
        state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def keys_for(question):
    """Answer keys in the order decide returns them (noul: true, false)."""
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["true", "false"]


def select(cands, reported, cap=PER_SOURCE):
    """Selection rules 1-3: drop reported states, drop repeats, keep the first cap.
    Returns (kept, n_excluded_reported, n_repeats)."""
    kept, seen, excluded, repeats = [], set(), 0, 0
    for c in cands:
        h = state_hash(c["state"])
        if h in reported:
            excluded += 1
        elif h in seen:
            repeats += 1
        else:
            seen.add(h)
            kept.append(c)
    return kept[:cap], excluded, repeats


def jevbench():
    """jevbench with its dataset cache redirected to WORK; the clone is never written."""
    root = WORK / "jevbench"
    (root / "data").mkdir(parents=True, exist_ok=True)
    for f in (BENCH / "data").glob("*.json"):
        if not (root / "data" / f.name).exists():
            shutil.copy2(f, root / "data" / f.name)
    sys.path.insert(0, str(BENCH))
    import jevbench as jb
    jb.ROOT = root
    return jb


def jevbench_candidates(jb, task, split):
    """The task's own constructor with its loader redirected to the held-out split."""
    orig_rows, orig_get = jb.hf_rows, jb._get

    def rows(dataset, config, _split, _n, seed=0, page=100):
        return orig_rows(dataset, config, split, DRAW, seed=HELDOUT_SEED, page=page)

    def get(url):
        return orig_get(url.replace("/banking_data/test.csv", "/banking_data/train.csv"))

    jb.hf_rows, jb._get = rows, get
    try:
        return jb.TASKS[task][0](DRAW)[0]
    finally:
        jb.hf_rows, jb._get = orig_rows, orig_get


def reported_hashes(jb):
    """R (spec §13.5): every state of a reported row."""
    hashes = set()
    for task in jb.TASKS:
        hashes |= {state_hash(e["state"]) for e in jb.TASKS[task][0](500)[0]}
    for model in REPORTED_MODELS:
        files = sorted((EVAL / model / "jevbench/raw").glob("*.jsonl"))
        if not files:
            sys.exit(f"no saved raw rows under {EVAL / model}: R would be incomplete")
        for f in files:
            hashes |= {state_hash(r["state"]) for r in read_jsonl(f)}
    hashes |= {state_hash(r["state"]) for r in calsets.load_synth(CAL / "data/val.jsonl")}
    for name in calsets.PUBLIC_N:
        hashes |= {state_hash(r["state"]) for r in calsets.public_records(name)}
    return hashes


def record_examples(recs):
    out = []
    for r in recs:
        q, keys, target = calsets.to_question(r)
        out.append({"state": r["state"], "question": q, "keys": keys, "gold": target.index(1.0)})
    return out


def candidates(jb):
    """[(source, type, split, seed, n drawn, candidate examples)] in build order."""
    out = []
    for task, split in JEVBENCH_SPLIT.items():
        q = jb.TASKS[task][0](500)[1]  # the reported question, criteria in reported order
        keys = keys_for(q)
        drawn = jevbench_candidates(jb, task, split)
        ex = [{"state": e["state"], "question": q, "keys": keys, "gold": keys.index(e["label"])}
              for e in drawn if e["label"] in keys]
        out.append((task, q["type"], split, HELDOUT_SEED, len(drawn), ex))
    sys.path.insert(0, str(CAL / "data"))
    import generate
    synth = generate.build_synthetic(SYNTH_TICKETS, 0.05, SYNTH_SEED)
    for qtype in ("choice", "score", "noul"):
        recs = [r for r in synth if r["type"] == qtype]
        out.append((f"synth-{qtype}", qtype, f"generate.py seed {SYNTH_SEED}", SYNTH_SEED, len(recs),
                    record_examples(recs)))
    for name in PUBLIC_TRAIN:
        recs = calsets.public_records(name, split="train", n=DRAW)
        out.append((name, "choice", "train", 1, DRAW, record_examples(recs)))
    return out


def cmd_build(_a):
    jb = jevbench()
    rep = reported_hashes(jb)
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / "examples").mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, qtype, split, seed, drawn, ex in candidates(jb):
        kept, excluded, repeats = select(ex, rep)
        assert not {state_hash(c["state"]) for c in kept} & rep, name
        path = WORK / "examples" / f"{name}.jsonl"
        with open(path, "w") as f:
            for c in kept:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        manifest[name] = {"type": qtype, "split": split, "seed": seed, "drawn": drawn,
                          "label_filtered": drawn - len(ex), "excluded_reported": excluded,
                          "repeats": repeats, "kept": len(kept), "sha256": sha256_file(path)}
        print(f"{name:<16} {qtype:<6} drawn {drawn:>4}  excluded {excluded:>3}  kept {len(kept)}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (OUT / "reported_state_sha256.txt").write_text("\n".join(sorted(rep)) + "\n")
    verify_dirs(OUT, WORK)


def verify_dirs(out, work):
    """No held-out example is a reported state, and every examples file is the one built."""
    rep = set((out / "reported_state_sha256.txt").read_text().split())
    manifest = json.loads((out / "manifest.json").read_text())
    total = 0
    for name, m in manifest.items():
        path = work / "examples" / f"{name}.jsonl"
        assert sha256_file(path) == m["sha256"], f"{name}: examples changed since build"
        rows = read_jsonl(path)
        assert len(rows) == m["kept"], f"{name}: {len(rows)} rows, manifest says {m['kept']}"
        hit = sum(state_hash(r["state"]) in rep for r in rows)
        assert hit == 0, f"{name}: {hit} held-out rows are reported rows"
        total += len(rows)
    print(f"verify: {total} held-out rows in {len(manifest)} sources, none among {len(rep)} reported states")


def cmd_verify(_a):
    verify_dirs(OUT, WORK)


def wait_health(port, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() > deadline:
            sys.exit(f"serve on :{port} not healthy after {timeout}s")
        time.sleep(2)


def answer_row(name, qtype, keys, gold, ans):
    probs = [ans["noul"], 1 - ans["noul"]] if qtype == "noul" else [ans["probabilities"][k] for k in keys]
    return {"source": name, "type": qtype, "probs": [float(f"{p:.12g}") for p in probs], "gold": gold}


def run_sources(manifest, named, allow_restricted=False):
    """Sources `run` asks, in manifest order. The default is the licence-clear
    ones only (licences.py); a restricted source must be named and opted into."""
    unknown = sorted(set(named or ()) - set(manifest))
    if unknown:
        sys.exit(f"not in the manifest: {unknown}")
    if not named:
        return [n for n in manifest if not licences.is_restricted(n)]
    bad = licences.restricted_in(named)
    if bad and not allow_restricted:
        sys.exit(f"restricted sources {bad} (bench/jev/DATA-LICENSES.md) need --allow-restricted")
    return [n for n in manifest if n in set(named)]


def cmd_run(a):
    manifest = json.loads((OUT / "manifest.json").read_text())
    out = Path(a.out)
    if out.name != "heldout":
        sys.exit("--out must be bench/jev/<model>/heldout")
    out.mkdir(parents=True, exist_ok=True)
    verify_dirs(OUT, WORK)
    wait_health(a.port)
    url = f"http://127.0.0.1:{a.port}/v1/systemone"
    for name in run_sources(manifest, a.sources, a.allow_restricted):
        rows = read_jsonl(WORK / "examples" / f"{name}.jsonl")
        path = out / f"{name}.jsonl"
        done = len(read_jsonl(path)) if path.exists() else 0  # resumable
        with open(path, "a") as f:
            for i in range(done, len(rows)):
                r = rows[i]
                body = json.dumps({"model": a.model, "state": r["state"], "questions": {"q": r["question"]}})
                req = urllib.request.Request(url, data=body.encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as resp:
                    ans = json.load(resp)["answers"]["q"]
                f.write(json.dumps(answer_row(name, manifest[name]["type"], r["keys"], r["gold"], ans)) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(f"{name}: {i}/{len(rows)}", file=sys.stderr, flush=True)
        print(f"{name}: {len(rows)} rows -> {path}", flush=True)
    print("done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("verify")
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, default=11435)
    r.add_argument("--model", required=True, help="model as the reported runs named it (the file path)")
    r.add_argument("--out", required=True, help="bench/jev/<model>/heldout")
    r.add_argument("--allow-restricted", action="store_true",
                   help="permit a named restricted source (licences.py); never the default")
    r.add_argument("sources", nargs="*", help="default: every licence-clear source in the manifest")
    a = ap.parse_args(argv)
    {"build": cmd_build, "verify": cmd_verify, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
