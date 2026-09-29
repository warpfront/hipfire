"""Decide calibration (spec §13): freeze the reported predictions, fit one
temperature per question type on held-out rows, apply it.

Rows everywhere are {"source", "type", "probs", "gold"}: the probabilities
as served (raw, T = 1) in answer-key order, gold an index into them. No text,
no keys (git stores these files zlib-compressed; the keys would dominate).

  freeze  --src ~/repos/hipfire-jev-eval/bench/jev/<model> --out bench/jev/<model>/probs
  fit     --heldout bench/jev/<model>/heldout --model-id <models.toml id>
          --build <sha> --out bench/jev/<model>/calibration.json

Some jev-bench tasks and calibration sets are frozen locally by `freeze` but
not committed to this repository: their source dataset's licence terms do
not clearly permit redistributing derived per-row data (labels and served
probabilities), even without the original text. See
bench/jev/DATA-LICENSES.md. `read_jsonl` and the report reader transparently
accept a `.jsonl.gz` in place of a `.jsonl` (the form committed rows take),
and silently skip a source with neither, so a repo checkout with only the
licence-clear subset still reports correctly.
"""
import argparse
import gzip
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

TYPES = ("choice", "score", "noul")
T_MIN, T_MAX = 0.05, 20.0
PER_SOURCE_CAP = 300
BOOTSTRAP = 100
MIN_NLL_GAIN = 0.01
SCORE_TASKS = {"yelp-stars", "sentiment-it"}
NOUL_TASKS = {"sms-spam", "duplicates", "doc-yesno", "offensive"}


def r12(x):
    """12 significant digits: log p stays exact to ~1e-12, files stay small."""
    return float(f"{x:.12g}")


def argmax(xs):
    return max(range(len(xs)), key=xs.__getitem__)


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rows(paths):
    rows = []
    for p in paths:
        for r in read_jsonl(p):
            if min(r["probs"]) <= 0:
                raise ValueError(f"{p}: a probability <= 0 has no label logit (spec §13.3)")
            rows.append(r)
    return rows


def apply_t(probs, t):
    """softmax(log p / t) == softmax(z / t), since log p = z - logsumexp(z)."""
    s = [math.log(p) / t for p in probs]
    m = max(s)
    e = [math.exp(x - m) for x in s]
    tot = sum(e)
    return [x / tot for x in e]


def ece(pairs, bins=10):
    """Top-probability ECE, binned exactly as jevbench.py's ece()."""
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        ins = [(p, c) for p, c in pairs if lo < p <= hi or (b == 0 and p == 0)]
        if ins:
            e += len(ins) / len(pairs) * abs(sum(p for p, _ in ins) / len(ins) - sum(c for _, c in ins) / len(ins))
    return e


def _prep(rows):
    return [([math.log(p) for p in r["probs"]], r["gold"]) for r in rows]


def _nll(data, t):
    tot = 0.0
    for lp, y in data:
        s = [x / t for x in lp]
        m = max(s)
        tot += m + math.log(sum(math.exp(x - m) for x in s)) - s[y]
    return tot / len(data)


def nll(rows, t):
    """Mean negative log-likelihood of the gold labels at temperature t."""
    return _nll(_prep(rows), t)


def _grad_hess(data, b):
    """dNLL/db and d2NLL/db2 at b = 1/T: mean(E_q[l] - l_y), mean(Var_q[l])."""
    g = h = 0.0
    for lp, y in data:
        s = [b * x for x in lp]
        m = max(s)
        w = [math.exp(x - m) for x in s]
        z = sum(w)
        mean = sum(wi * x for wi, x in zip(w, lp)) / z
        g += mean - lp[y]
        h += sum(wi * (x - mean) ** 2 for wi, x in zip(w, lp)) / z
    return g / len(data), h / len(data)


def _fit(data, strict=True):
    """argmin over T in [T_MIN, T_MAX] of NLL. NLL is convex in b = 1/T, so
    safeguarded Newton on its monotone derivative. strict: a bound is an error."""
    lo, hi = 1.0 / T_MAX, 1.0 / T_MIN
    b = 1.0
    for _ in range(200):
        g, h = _grad_hess(data, b)
        if g > 0:
            hi = b
        else:
            lo = b
        nb = b - g / h if h > 0 else (lo + hi) / 2
        if not lo < nb < hi:
            nb = (lo + hi) / 2
        if abs(nb - b) <= 1e-12 * b:
            b = nb
            break
        b = nb
    t = 1.0 / b
    if strict and (t <= T_MIN * (1 + 1e-6) or t >= T_MAX * (1 - 1e-6)):
        raise ValueError(f"fitted T={t:.6g} is at the search bound [{T_MIN}, {T_MAX}]")
    return t


def fit_temperature(rows):
    return _fit(_prep(rows))


def bootstrap_ci(data, n=BOOTSTRAP, seed=0):
    """95% interval of T over n resamples (a resample at a bound counts at the bound)."""
    rng = random.Random(seed)
    ts = sorted(_fit(rng.choices(data, k=len(data)), strict=False) for _ in range(n))
    return ts[int(0.025 * n)], ts[int(0.975 * n) - 1]


def pool(rows, qtype):
    """Rows of one type, at most PER_SOURCE_CAP per source (first ones)."""
    by = defaultdict(list)
    for r in rows:
        if r["type"] == qtype:
            by[r["source"]].append(r)
    return {s: rs[:PER_SOURCE_CAP] for s, rs in sorted(by.items())}


def fit_types(rows):
    """Per type: T, its 95% interval, held-out NLL raw / calibrated, ship rule (spec §13.4)."""
    out = {}
    for qtype in TYPES:
        srcs = pool(rows, qtype)
        data = _prep([r for rs in srcs.values() for r in rs])
        if not data:
            continue
        entry = {"n": len(data), "sources": {s: len(v) for s, v in srcs.items()}}
        try:
            t = round(_fit(data), 3)
        except ValueError as e:
            out[qtype] = {**entry, "error": str(e), "ship": False}
            continue
        lo, hi = bootstrap_ci(data)
        raw, cal = _nll(data, 1.0), _nll(data, t)
        out[qtype] = {**entry, "t": t, "t_ci95": [round(lo, 3), round(hi, 3)], "nll_raw": raw,
                      "nll_cal": cal, "ship": (raw - cal) / raw >= MIN_NLL_GAIN and not lo <= 1.0 <= hi}
    return out


def summarize(rows, temps):
    """Accuracy, ECE and NLL raw vs calibrated at temps ({type: T}, missing = 1);
    for score rows also mean |E[score] - gold|. Asserts no argmax moved."""
    raw, cal = [], []
    nll_raw = nll_cal = mae_raw = mae_cal = 0.0
    n_score = 0
    for r in rows:
        p = r["probs"]
        q = apply_t(p, temps.get(r["type"], 1.0))
        top = argmax(p)
        if argmax(q) != top:
            raise AssertionError(f"{r['source']}: calibration moved an argmax")
        c = int(top == r["gold"])
        raw.append((p[top], c))
        cal.append((q[top], c))
        nll_raw -= math.log(p[r["gold"]])
        nll_cal -= math.log(q[r["gold"]])
        if r["type"] == "score":
            n_score += 1
            mae_raw += abs(sum(i * x for i, x in enumerate(p)) - r["gold"])
            mae_cal += abs(sum(i * x for i, x in enumerate(q)) - r["gold"])
    n = len(rows)
    out = {"n": n, "acc": sum(c for _, c in raw) / n, "ece_raw": ece(raw), "ece_cal": ece(cal),
           "nll_raw": nll_raw / n, "nll_cal": nll_cal / n}
    if n_score:
        out["mae_raw"], out["mae_cal"] = mae_raw / n_score, mae_cal / n_score
    return out


def slim_bench_row(task, r):
    """A jevbench raw/ row (full `prob` distribution) without its text."""
    keys = list(r["prob"])
    qtype = "noul" if task in NOUL_TASKS else "score" if task in SCORE_TASKS else "choice"
    return {"source": task, "type": qtype, "probs": [r12(r["prob"][k]) for k in keys],
            "gold": keys.index(r["label"])}


def slim_cal_row(name, r):
    """A run_calibration.py row without its usage/confidence fields."""
    keys = r["option_keys"]
    return {"source": name, "type": r["type"], "probs": [r12(p) for p in r["probs"]],
            "gold": keys.index(r["gold"])}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def find_rows(out, prefix):
    """{name: path} for <prefix><name>.jsonl or .jsonl.gz under `out`, matching
    calibrate.py freeze's own naming. A committed repo may hold only the
    `.jsonl.gz` for a source (licence-clear ones, §DATA-LICENSES.md); a full
    local freeze holds the plain `.jsonl` for every source. When both exist,
    the `.gz` (the committed one) wins."""
    found = {}
    for p in sorted(Path(out).glob(f"{prefix}*.jsonl")) + sorted(Path(out).glob(f"{prefix}*.jsonl.gz")):
        name = p.name.removeprefix(prefix).removesuffix(".gz").removesuffix(".jsonl")
        found[name] = p
    return found


def cmd_freeze(a):
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for p in sorted((src / "jevbench/raw").glob("*.jsonl")):
        if p.stem.startswith("x-"):
            continue  # the experiments are not in the report tables
        rows = [slim_bench_row(p.stem, r) for r in read_jsonl(p)]
        pred_path = src / "jevbench/predictions" / f"{p.stem}.jsonl"
        preds = read_jsonl(pred_path)
        if len(preds) != len(rows):
            sys.exit(f"{p.stem}: {len(rows)} raw rows vs {len(preds)} saved predictions")
        for r, pr in zip(rows, preds):
            top = argmax(r["probs"])
            if abs(r["probs"][top] - pr["p_top"]) > 5e-5 or int(top == r["gold"]) != pr["correct"]:
                sys.exit(f"{p.stem} row {pr['i']}: raw row does not match the saved prediction")
        write_rows(out / f"jevbench_{p.stem}.jsonl", rows)
        sources[str(p)] = sha256_file(p)
        sources[str(pred_path)] = sha256_file(pred_path)
    for p in sorted((src / "calibration").glob("hipfire_*.jsonl")):
        name = p.stem.removeprefix("hipfire_")
        write_rows(out / f"cal_{name}.jsonl", [slim_cal_row(name, r) for r in read_jsonl(p)])
        sources[str(p)] = sha256_file(p)
    (out / "SOURCES.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    print(f"froze {len(sources)} source files -> {out}")


def toml_snippet(model_id, temps):
    if not temps:
        return f"# {model_id}: no question type passed the ship rule; leave it uncalibrated"
    body = "\n".join(f"{k} = {v}" for k, v in temps.items())
    return f'[models."{model_id}".overrides.decide.calibration]\n{body}'


def cmd_fit(a):
    src = Path(a.heldout)
    if src.name != "heldout":
        sys.exit("fit reads only a heldout/ directory: reported rows never feed the fit (spec §13.4)")
    rows = load_rows(sorted(src.glob("*.jsonl")))
    types = fit_types(rows)
    ship = {t: v["t"] for t, v in types.items() if v["ship"]}
    doc = {"model_id": a.model_id, "build": a.build, "types": types, "decide_calibration": ship}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(toml_snippet(a.model_id, ship))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--src", required=True)
    f.add_argument("--out", required=True)
    g = sub.add_parser("fit")
    g.add_argument("--heldout", required=True)
    g.add_argument("--model-id", required=True)
    g.add_argument("--build", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"freeze": cmd_freeze, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
