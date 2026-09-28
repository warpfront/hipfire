"""Run the scienthoon/jev-ood-calibration sets against local hipfire serve,
writing rows in that repo's result schema (type, option_keys, probs, target,
pred, gold, confidence, source, usage)."""
import argparse
import json
import os
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

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
NOUL_TRUE, NOUL_FALSE = "Yes, the statement is true.", "No, the statement is false."


def hf_rows(ds, cfg, n):
    base = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds, safe='')}"
            f"&config={cfg}&split=validation")
    rows = []
    for off in range(0, n, 100):
        with urllib.request.urlopen(base + f"&offset={off}&length={min(100, n - off)}", timeout=60) as r:
            rows += [x["row"] for x in json.load(r)["rows"]]
    return rows


def public_records(name):
    # Mirrors convert.py: first ≤2000 validation rows, filter, Random(1).shuffle
    # (verified row-for-row against Jev's committed results on 2026-09-28).
    if name == "openbookqa":
        src = hf_rows("allenai/openbookqa", "main", 500)
        recs = [{"state": s["question_stem"], "type": "choice", "question": "Which option is the correct answer?",
                 "options": dict(zip(s["choices"]["label"], s["choices"]["text"])), "label": str(s["answerKey"])}
                for s in src]
    elif name == "commonsense_qa":
        src = hf_rows("tau/commonsense_qa", "default", 1221)
        recs = [{"state": s["question"], "type": "choice", "question": "Which option is the correct answer?",
                 "options": dict(zip(s["choices"]["label"], s["choices"]["text"])), "label": str(s["answerKey"])}
                for s in src]
    else:
        src = hf_rows("Rowan/hellaswag", "default", 2000)
        recs = [{"state": s["ctx"], "type": "choice", "question": "Which ending most plausibly continues the text?",
                 "options": {str(i): e for i, e in enumerate(s["endings"])}, "label": str(s["label"]).strip()}
                for s in src]
    recs = [r for r in recs if r["label"] in r["options"]]
    random.Random(1).shuffle(recs)
    for r in recs:
        r["source"] = name
    return recs


def to_question(r):
    if r["type"] == "choice":
        keys = list(r["options"])
        return ({"type": "choice", "instructions": r["question"], "criteria": r["options"]},
                keys, [1.0 if k == r["label"] else 0.0 for k in keys])
    if r["type"] == "score":
        keys = [str(i) for i in range(len(r["levels"]))]
        return ({"type": "score", "instructions": r["question"], "criteria": [str(x) for x in r["levels"]]},
                keys, [1.0 if i == int(r["label"]) else 0.0 for i in range(len(keys))])
    return ({"type": "noul", "instructions": r["question"], "criteria": {"true": NOUL_TRUE, "false": NOUL_FALSE}},
            ["yes", "no"], [1.0, 0.0] if r["label"] is True else [0.0, 1.0])


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
    recs = ([json.loads(l) for l in open(cal / "data/val.jsonl")] if s == "synth" else public_records(s))
    path = out / f"hipfire_{s}.jsonl"
    with open(path, "w") as f:
        for i, r in enumerate(recs):
            f.write(json.dumps(ask(r)) + "\n")
            if i % 100 == 0:
                print(f"{s}: {i}/{len(recs)}", file=sys.stderr)
    print(f"wrote {path}")
