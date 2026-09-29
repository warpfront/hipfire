"""Record builders shared by run_calibration.py and heldout.py: the
scienthoon/jev-ood-calibration sets (synthetic tickets, public QA) and their
decide questions. The validation rebuild mirrors that repo's convert.py
(verified row-for-row against Jev's committed results on 2026-09-28)."""
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request

NOUL_TRUE, NOUL_FALSE = "Yes, the statement is true.", "No, the statement is false."
# Rows read per public set for the reported (validation) rebuild.
PUBLIC_N = {"openbookqa": 500, "commonsense_qa": 1221, "hellaswag": 2000}


def _get_json(url):
    """GET with patience: datasets-server rate-limits bursts and 5xxs at times."""
    for i in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError):
            if i == 7:
                raise
            time.sleep(min(60, 3 * 2 ** i))


def hf_rows(ds, cfg, n, split="validation"):
    base = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds, safe='')}"
            f"&config={cfg}&split={split}")
    rows = []
    for off in range(0, n, 100):
        rows += [x["row"] for x in _get_json(base + f"&offset={off}&length={min(100, n - off)}")["rows"]]
    return rows


def public_records(name, split="validation", n=None):
    """The first n rows of `split`, filtered, then random.Random(1).shuffle.
    With the defaults: exactly the reported validation rows."""
    n = PUBLIC_N[name] if n is None else n
    if name == "openbookqa":
        src = hf_rows("allenai/openbookqa", "main", n, split)
        recs = [{"state": s["question_stem"], "type": "choice", "question": "Which option is the correct answer?",
                 "options": dict(zip(s["choices"]["label"], s["choices"]["text"])), "label": str(s["answerKey"])}
                for s in src]
    elif name == "commonsense_qa":
        src = hf_rows("tau/commonsense_qa", "default", n, split)
        recs = [{"state": s["question"], "type": "choice", "question": "Which option is the correct answer?",
                 "options": dict(zip(s["choices"]["label"], s["choices"]["text"])), "label": str(s["answerKey"])}
                for s in src]
    else:
        src = hf_rows("Rowan/hellaswag", "default", n, split)
        recs = [{"state": s["ctx"], "type": "choice", "question": "Which ending most plausibly continues the text?",
                 "options": {str(i): e for i, e in enumerate(s["endings"])}, "label": str(s["label"]).strip()}
                for s in src]
    recs = [r for r in recs if r["label"] in r["options"]]
    random.Random(1).shuffle(recs)
    for r in recs:
        r["source"] = name
    return recs


def load_synth(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_question(r):
    """(decide question, option keys, one-hot target) for one record."""
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
