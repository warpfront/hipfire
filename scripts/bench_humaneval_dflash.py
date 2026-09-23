#!/usr/bin/env python3
# bench_humaneval_dflash.py — run dflash_spec_demo across a HumanEval sample.
# HumanEval prompts are code-continuation (signature + docstring) so τ stays
# in code-token regime. Matches Lucebox methodology.
#
# Usage:
#   python3 scripts/bench_humaneval_dflash.py            # default: 33 prompts, DFlash mode
#   python3 scripts/bench_humaneval_dflash.py --ar       # AR baseline mode (no draft used)
#   python3 scripts/bench_humaneval_dflash.py --n 25     # sample size
#   python3 scripts/bench_humaneval_dflash.py --jsonl PATH --max 128

import argparse, json, os, re, subprocess, sys, time

def pct(sorted_arr, p):
    if not sorted_arr:
        return float("nan")
    idx = int(round((len(sorted_arr) - 1) * p / 100))
    return sorted_arr[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/tmp/humaneval/HumanEval.jsonl")
    ap.add_argument("--target", default=os.path.expanduser("~/.hipfire/models/qwen3.5-27b.mq4"))
    ap.add_argument("--draft", default=os.path.expanduser("~/.hipfire/models/qwen35-27b-dflash-mq4.hfq"))
    ap.add_argument("--demo", default="./target/release/examples/dflash_spec_demo")
    ap.add_argument("--max", type=int, default=128, help="max tokens per prompt")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n", type=int, default=33, help="sample size (0..164)")
    ap.add_argument("--kv-mode", default="q8", help="q8 | asym3 | asym4 | asym2")
    ap.add_argument("--ar", action="store_true", help="AR baseline mode (passes --ar-baseline; no spec decode)")
    ap.add_argument("--ddtree-batched", action="store_true", help="enable --ddtree-batched")
    ap.add_argument("--ddtree-budget", type=int, default=None)
    ap.add_argument("--ddtree-topk", type=int, default=None)
    ap.add_argument("--label", default="", help="label printed in the summary (e.g. 'linear-asym3')")
    args = ap.parse_args()
    if args.ar and (args.ddtree_batched or args.ddtree_budget is not None or args.ddtree_topk is not None):
        print("--ar is incompatible with --ddtree-* flags", file=sys.stderr); sys.exit(2)

    with open(args.jsonl) as f:
        tasks = [json.loads(l) for l in f]
    if not tasks:
        print("empty jsonl", file=sys.stderr); sys.exit(1)

    stride = max(1, len(tasks) // args.n)
    sampled = tasks[::stride][: args.n]
    print(f"# sampling {len(sampled)} of {len(tasks)} HumanEval prompts (stride={stride})")
    print(f"# target: {args.target}")
    print(f"# draft:  {args.draft}")
    extra_args = ["--kv-mode", args.kv_mode]
    if args.ar:
        extra_args.append("--ar-baseline")
    if args.ddtree_batched:
        extra_args.append("--ddtree-batched")
    if args.ddtree_budget is not None:
        extra_args += ["--ddtree-budget", str(args.ddtree_budget)]
    if args.ddtree_topk is not None:
        extra_args += ["--ddtree-topk", str(args.ddtree_topk)]
    label = args.label or ("ar" if args.ar else "ddtree" if args.ddtree_batched else "linear")
    print(f"# label={label}  max={args.max} ctx={args.ctx} --no-chatml kv={args.kv_mode}  "
          f"extra={' '.join(extra_args)}")
    print()
    print(f"{'task_id':<16} {'dec_t/s':>8} {'tau':>7} {'pre_t/s':>8} {'ttft_ms':>8} {'emit':>5} {'ptok':>5} {'run_s':>6}")
    print("-" * 76)

    results = []
    for task in sampled:
        tid = task["task_id"]
        prompt = task["prompt"]
        t0 = time.time()
        try:
            proc = subprocess.run(
                [args.demo, "--target", args.target, "--draft", args.draft,
                 "--prompt", prompt, "--max", str(args.max), "--ctx", str(args.ctx),
                 "--no-chatml", *extra_args],
                capture_output=True, text=True, timeout=240)
            out = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            print(f"{tid:<16} TIMEOUT")
            continue
        dt = time.time() - t0

        m_toks    = re.search(r"emitted: (\d+) tokens in ([\d.]+)s\s+\(([\d.]+) tok/s\)", out)
        m_tau     = re.search(r"\xcf\x84=([\d.]+)|τ=([\d.]+)", out)
        m_cyc     = re.search(r"cycles: (\d+)", out)
        m_acc     = re.search(r"accepted: (\d+)", out)
        m_pre_ts  = re.search(r"prefill_tok_s:\s*([\d.]+)", out)
        m_pre_sec = re.search(r"prefill_secs:\s*([\d.]+)", out)
        m_ttft    = re.search(r"ttft_ms:\s*([\d.]+)", out)
        m_ptok    = re.search(r"prompt_tokens:\s*(\d+)", out)

        # In AR mode there is no τ / cycles / accepted — only emitted+tok_s are required.
        if not m_toks or (not args.ar and not m_tau):
            print(f"{tid:<16} PARSE_FAIL  run_s={dt:.1f}")
            results.append({"tid": tid, "tok_s": None})
            continue

        r = {
            "tid": tid,
            "emitted":  int(m_toks.group(1)),
            "tok_s":    float(m_toks.group(3)),
            "tau":      float(m_tau.group(1) or m_tau.group(2)) if m_tau else float("nan"),
            "cyc":      int(m_cyc.group(1)) if m_cyc else -1,
            "acc":      int(m_acc.group(1)) if m_acc else -1,
            "pre_ts":   float(m_pre_ts.group(1))  if m_pre_ts  else float("nan"),
            "pre_sec":  float(m_pre_sec.group(1)) if m_pre_sec else float("nan"),
            "ttft_ms":  float(m_ttft.group(1))    if m_ttft    else float("nan"),
            "ptok":     int(m_ptok.group(1))      if m_ptok    else -1,
            "run_s":    dt,
        }

        print(f"{tid:<16} {r['tok_s']:8.2f} {r['tau']:7.3f} {r['pre_ts']:8.1f} "
              f"{r['ttft_ms']:8.1f} {r['emitted']:5d} {r['ptok']:5d} {dt:6.1f}")
        results.append(r)

    # summary
    valid = [r for r in results if r.get("tok_s") is not None]
    if not valid:
        print("\nno valid results"); sys.exit(2)

    def stats(name, key, fmt="7.2f"):
        xs = sorted(r[key] for r in valid if r[key] == r[key])  # filter NaN
        if not xs:
            print(f"{name}  (no data)"); return
        mean = sum(xs) / len(xs)
        print(f"{name:<10} mean={mean:{fmt}}  median={pct(xs,50):{fmt}}  "
              f"p10={pct(xs,10):{fmt}}  p90={pct(xs,90):{fmt}}  "
              f"min={xs[0]:{fmt}}  max={xs[-1]:{fmt}}")

    print()
    print(f"=== SUMMARY (n={len(valid)}/{len(sampled)}) ===")
    stats("dec tok/s", "tok_s")
    if not args.ar:
        stats("tau",       "tau", fmt="7.3f")
    stats("pre tok/s", "pre_ts")
    stats("ttft_ms",   "ttft_ms", fmt="8.1f")

if __name__ == "__main__":
    main()
