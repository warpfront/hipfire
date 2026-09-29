# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""Run the scienthoon/jev-ood-calibration sets against local hipfire serve,
writing rows in that repo's result schema (type, option_keys, probs, target,
pred, gold, confidence, source, usage)."""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from calsets import load_synth, public_records, to_question

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=11435)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("sets", nargs="+", choices=["synth", "openbookqa", "commonsense_qa", "hellaswag"])
a = ap.parse_args()
cal = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
URL = f"http://127.0.0.1:{a.port}/v1/systemone"


def ask(r):
    q, keys, target = to_question(r)
    body = json.dumps({"model": a.model, "state": r["state"], "questions": {"q": q}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        v = json.load(resp)
    ans = v["answers"]["q"]
    probs = [ans["noul"], 1 - ans["noul"]] if r["type"] == "noul" else [ans["probabilities"][k] for k in keys]
    pred = keys[max(range(len(probs)), key=probs.__getitem__)]
    gold = keys[max(range(len(target)), key=target.__getitem__)]
    return {"type": r["type"], "option_keys": keys, "probs": probs, "target": target, "pred": pred,
            "gold": gold, "confidence": ans.get("confidence"), "source": r.get("source"),
            "usage": {"inputTokens": v["usage"]["input_tokens"], "outputTokens": 0}}


for s in a.sets:
    recs = load_synth(cal / "data/val.jsonl") if s == "synth" else public_records(s)
    path = out / f"hipfire_{s}.jsonl"
    with open(path, "w") as f:
        for i, r in enumerate(recs):
            f.write(json.dumps(ask(r)) + "\n")
            if i % 100 == 0:
                print(f"{s}: {i}/{len(recs)}", file=sys.stderr)
    print(f"wrote {path}")
