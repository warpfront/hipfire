#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Packed RTZ+FMA end-to-end A/B orchestrator (developer-only, isolated scratch).
# Parent runs this via hub. This script is NOT executed by its author.
#
# Two SEPARATE routes (never cross-compared):
#   1. demo    — dflash_spec_demo, historical merge-sort flags, ABBAABBA (4 each)
#   2. product — hipfire bench --spec dflash noslots/stateless, 3 fresh procs/arm
#
# Baseline bins: .scratch/baseline-bin/{hipfire,daemon,dflash_spec_demo}
# Candidate bins: target/release/{hipfire,daemon} + examples/dflash_spec_demo
# Baseline CLI → baseline daemon; candidate CLI → candidate daemon.
# No compile, no registry fetch, no cache overwrite, no clock changes.
# Fail-fast on nonzero; no autofallback.
set -euo pipefail

REPO="$HOME/xtx-gfx1100-baseline"
PROMPT="$REPO/benchmarks/prompts/merge_sort_thinking_off.txt"
PROMPT_MD5="253c7ac50857fe6d0e10fb0d2c5e35c0"
HWGATE_LOCK="$HOME/actions-runner/_cache/hw-gate-gpu.lock"

# Demo fixture (byte-identical to run-xtx-demo-hist.sh / fuse-integ.sh)
DEMO_TARGET="/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt"
DEMO_DRAFT="/home/kaden/.hipfire/models/qwen38-27b-dflash-mq4.hfq"

# Product bench fixture (path form; draft explicit — no registry fetch)
PRODUCT_MODEL_TAG="qwen3.8:27b-mq4-xt"
PRODUCT_MODEL="$HOME/.hipfire/models/qwen3.8-27b.mq4-xt"
MODELS_DIR="/home/kaden/.hipfire/models"
PRODUCT_MODEL_BYTES="14980361216"
PRODUCT_DRAFT="/home/kaden/.hipfire/models/qwen38-27b-dflash-mq4.hfq"

BASE_BIN="$REPO/.scratch/stage2-baseline-bin"
CAND_BIN_DIR="$REPO/target/release"
CAND_DEMO="$CAND_BIN_DIR/examples/dflash_spec_demo"

BASE_HIPFIRE="$BASE_BIN/hipfire"
BASE_DAEMON="$BASE_BIN/daemon"
BASE_DEMO="$BASE_BIN/dflash_spec_demo"
CAND_HIPFIRE="$CAND_BIN_DIR/hipfire"
CAND_DAEMON="$CAND_BIN_DIR/daemon"

RUN_DIR="$REPO/.scratch/packed-stage2-ab-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"/{demo,product,hashes,meta}
LOG="$RUN_DIR/run.log"
# Line-buffered progress so READY is visible before long GPU work.
exec > >(stdbuf -oL -eL tee -a "$LOG") 2>&1

die() { echo "[packed-ab] FATAL: $*" >&2; exit 1; }
progress() { echo "[packed-ab] $*"; }

# --- live process guards ---
if pgrep -x dflash_spec_demo >/dev/null; then die "DEMO_LIVE_ABORT"; fi
if pgrep -x daemon >/dev/null; then die "DAEMON_LIVE_ABORT"; fi

# --- provenance guards (file checks only; no GPU, no builds) ---
test -x "$BASE_HIPFIRE" || die "missing baseline hipfire: $BASE_HIPFIRE"
test -x "$BASE_DAEMON"  || die "missing baseline daemon: $BASE_DAEMON"
test -x "$BASE_DEMO"    || die "missing baseline demo: $BASE_DEMO"
test -x "$CAND_HIPFIRE" || die "missing candidate hipfire: $CAND_HIPFIRE"
test -x "$CAND_DAEMON"  || die "missing candidate daemon: $CAND_DAEMON"
test -x "$CAND_DEMO"    || die "missing candidate demo: $CAND_DEMO"
test -f "$DEMO_TARGET"  || die "missing demo target: $DEMO_TARGET"
test -f "$DEMO_DRAFT"   || die "missing demo draft: $DEMO_DRAFT"
test -f "$PRODUCT_MODEL" || die "missing product model: $PRODUCT_MODEL"
test -f "$PRODUCT_DRAFT" || die "missing product draft: $PRODUCT_DRAFT"
test -f "$PROMPT" || die "missing prompt: $PROMPT"
test "$(md5sum "$PROMPT" | awk '{print $1}')" = "$PROMPT_MD5" || die "prompt md5 drift (want $PROMPT_MD5)"
test "$(stat -c%s "$PRODUCT_MODEL")" = "$PRODUCT_MODEL_BYTES" || die "product model size drift"

HEAD="$(git -C "$REPO" rev-parse HEAD)"
progress "head=$HEAD"

# --- hash every binary + prompt (preserve under hashes/) ---
{
  echo "# packed-ab provenance hashes $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head $HEAD"
  md5sum \
    "$BASE_HIPFIRE" "$BASE_DAEMON" "$BASE_DEMO" \
    "$CAND_HIPFIRE" "$CAND_DAEMON" "$CAND_DEMO" \
    "$PROMPT" "$DEMO_TARGET" "$DEMO_DRAFT" "$PRODUCT_MODEL" "$PRODUCT_DRAFT"
  echo "--- sha256 ---"
  sha256sum \
    "$BASE_HIPFIRE" "$BASE_DAEMON" "$BASE_DEMO" \
    "$CAND_HIPFIRE" "$CAND_DAEMON" "$CAND_DEMO" \
    "$PROMPT"
} | tee "$RUN_DIR/hashes/all.txt"

: > "$RUN_DIR/hashes/labeled.md5"
: > "$RUN_DIR/hashes/labeled.sha256"
hash_one() {
  local path=$1 label=$2
  md5sum "$path" | awk -v l="$label" '{print l, $1}' >> "$RUN_DIR/hashes/labeled.md5"
  sha256sum "$path" | awk -v l="$label" '{print l, $1}' >> "$RUN_DIR/hashes/labeled.sha256"
}
hash_one "$BASE_HIPFIRE" baseline-hipfire
hash_one "$BASE_DAEMON"  baseline-daemon
hash_one "$BASE_DEMO"    baseline-dflash_spec_demo
hash_one "$CAND_HIPFIRE" candidate-hipfire
hash_one "$CAND_DAEMON"  candidate-daemon
hash_one "$CAND_DEMO"    candidate-dflash_spec_demo
hash_one "$PROMPT"       prompt-merge_sort_thinking_off

# Record arm→bin map
cat > "$RUN_DIR/meta/bins.json" <<EOF
{
  "git_head": "$HEAD",
  "run_dir": "$RUN_DIR",
  "prompt": "$PROMPT",
  "prompt_md5": "$PROMPT_MD5",
  "demo": {
    "target": "$DEMO_TARGET",
    "draft": "$DEMO_DRAFT",
    "baseline_bin": "$BASE_DEMO",
    "candidate_bin": "$CAND_DEMO",
    "flags": ["--max","256","--temp","0.0","--no-chatml","--kv-mode","q8","--ctx","4096","--no-adaptive-b"],
    "env": {"HIPFIRE_VERIFY_GRAPH": "0"},
    "order": "ABBAABBA",
    "runs_per_arm": 4
  },
  "product": {
    "model": "$PRODUCT_MODEL",
    "draft": "$PRODUCT_DRAFT",
    "baseline_cli": "$BASE_HIPFIRE",
    "baseline_daemon": "$BASE_DAEMON",
    "candidate_cli": "$CAND_HIPFIRE",
    "candidate_daemon": "$CAND_DAEMON",
    "flags": ["bench","--spec","dflash","--runs","5","--warmups","3","--max-tokens","256","--backend","noslots","--workload","stateless","--kv-mode","q8","--json","--prompt-file"],
    "env": {"HIPFIRE_VERIFY_GRAPH": "0", "HIPFIRE_NO_REGISTRY_FETCH": "1", "HIPFIRE_DFLASH_DRAFT": "$PRODUCT_DRAFT"},
    "order": "ABBAAB",
    "runs_per_arm": 3,
    "note": "baseline CLI points baseline daemon; candidate CLI points candidate daemon"
  },
  "routes_isolated": true,
  "never_compare_demo_vs_product": true
}
EOF

# Drop ambient HIPFIRE_* that could leak identity across arms (except path/lock).
# Kernel cache is NOT redirected/overwritten — existing shared kernels are read-only OK.
while IFS= read -r line; do
  var="${line%%=*}"
  case "$var" in
    HIPFIRE_GPU_LOCKFILE|HIPFIRE_GPU_LOCK_OWNER) ;;
    HIPFIRE_*) unset "$var" || true ;;
  esac
done < <(env | grep -E '^HIPFIRE_' || true)
unset HIPFIRE_HOME || true
unset HIPFIRE_SERVE_ALLOW_INCOHERENT || true
unset HIPFIRE_DAEMON_BIN || true
unset HIPFIRE_CLI_BIN || true
unset HIPFIRE_KERNEL_CACHE || true

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HIPFIRE_VERIFY_GRAPH="0"
export HIPFIRE_NO_REGISTRY_FETCH="1"

# --- locks: agent GPU lock + hw-gate SHARED lock — once for both routes ---
# shellcheck disable=SC1091
source "$REPO/scripts/gpu-lock.sh"
gpu_acquire "packed-ab-rtz-fma"
exec 9>"$HWGATE_LOCK"
flock --shared --timeout 3600 9

# READY must appear before any GPU work; flush via stdbuf tee above.
echo "[packed-ab] READY run_dir=$RUN_DIR head=$HEAD"
echo "[packed-ab] routes=demo(ABBAABBA x dflash_spec_demo) product(ABBAAB x hipfire-bench) — never cross-compared"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
parse_demo_metrics() {
  # $1=stderr path → key=value lines for summary scrape
  local err=$1
  if [[ ! -f "$err" ]]; then echo "parse_error=missing_err"; return 0; fi
  local tok_s tau emitted cycles accept token_sha token_sha8
  tok_s=$(grep -E '^decode_tok_s:' "$err" | tail -1 | awk '{print $2}' || true)
  tau=$(grep -E '^decode_tau:' "$err" | tail -1 | awk '{print $2}' || true)
  emitted=$(grep -E '^decode_tokens_emitted:' "$err" | tail -1 | awk '{print $2}' || true)
  cycles=$(grep -E '^cycles:' "$err" | head -1 | awk '{print $2}' || true)
  accept=$(grep -E '^decode_accept_rate:' "$err" | tail -1 | awk '{print $2}' || true)
  if grep -q '^DFlash tokens:' "$err"; then
    token_sha=$(grep '^DFlash tokens:' "$err" | sha256sum | awk '{print $1}')
    token_sha8=$(printf '%s' "$token_sha" | cut -c1-8)
  else
    token_sha=""
    token_sha8=""
  fi
  printf 'decode_tok_s=%s\n' "${tok_s:-}"
  printf 'decode_tau=%s\n' "${tau:-}"
  printf 'decode_tokens_emitted=%s\n' "${emitted:-}"
  printf 'cycles=%s\n' "${cycles:-}"
  printf 'decode_accept_rate=%s\n' "${accept:-}"
  printf 'token_sha256=%s\n' "${token_sha:-}"
  printf 'token_sha8=%s\n' "${token_sha8:-}"
}

run_demo_once() {
  # $1=arm baseline|candidate  $2=slot_index  $3=order_index
  local arm=$1 slot=$2 order_i=$3
  local demo_bin out_dir home rc
  if [[ "$arm" == "baseline" ]]; then
    demo_bin="$BASE_DEMO"
  else
    demo_bin="$CAND_DEMO"
  fi
  out_dir="$RUN_DIR/demo/${arm}-${slot}"
  mkdir -p "$out_dir"
  progress "DEMO START arm=$arm slot=$slot order=$order_i bin=$demo_bin"

  # Isolated per-run home so we never clobber user ~/.hipfire config/cache writes.
  home="$out_dir/hipfire-home"
  write_product_home_config "$home"

  {
    echo "arm=$arm"
    echo "slot=$slot"
    echo "order_index=$order_i"
    echo "bin=$demo_bin"
    echo "bin_md5=$(md5sum "$demo_bin" | awk '{print $1}')"
    echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'argv='
    printf ' %q' "$demo_bin" \
      --target "$DEMO_TARGET" --draft "$DEMO_DRAFT" --prompt-file "$PROMPT" \
      --max 256 --temp 0.0 --no-chatml --kv-mode q8 --ctx 4096 --no-adaptive-b
    printf '\n'
  } > "$out_dir/command.txt"

  # Fresh process. Fail-fast on nonzero. Preserve host ROCm env; pin hipfire identity.
  set +e
  HIPFIRE_HOME="$home/.hipfire" \
  HIPFIRE_VERIFY_GRAPH=0 \
  HIPFIRE_NO_REGISTRY_FETCH=1 \
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  env -u HIPFIRE_DAEMON_BIN -u HIPFIRE_CLI_BIN -u HIPFIRE_KERNEL_CACHE -u HIPFIRE_DFLASH_DRAFT \
    "$demo_bin" \
      --target "$DEMO_TARGET" \
      --draft "$DEMO_DRAFT" \
      --prompt-file "$PROMPT" \
      --max 256 \
      --temp 0.0 \
      --no-chatml \
      --kv-mode q8 \
      --ctx 4096 \
      --no-adaptive-b \
    >"$out_dir/stdout" 2>"$out_dir/stderr"
  rc=$?
  set -e
  echo "$rc" > "$out_dir/exit_status"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out_dir/end.iso"
  parse_demo_metrics "$out_dir/stderr" | tee "$out_dir/metrics.txt"
  {
    echo "rc=$rc"
    cat "$out_dir/metrics.txt"
  } > "$out_dir/status.txt"

  progress "DEMO DONE arm=$arm slot=$slot rc=$rc $(tr '\n' ' ' < "$out_dir/metrics.txt")"
  if [[ "$rc" -ne 0 ]]; then
    echo "[packed-ab] DEMO STDERR TAIL:" >&2
    tail -40 "$out_dir/stderr" >&2 || true
    die "demo arm=$arm slot=$slot exited rc=$rc (no autofallback)"
  fi
}


write_product_home_config() {
  local home=$1
  mkdir -p "$home"
  cat > "$home/config.toml" <<CFG
schema_version = 1

[speculation]
dflash = "on"
mtp = "off"

[developer]
dflash_draft = "$PRODUCT_DRAFT"
CFG
}

run_product_once() {
  # $1=arm baseline|candidate  $2=slot_index  $3=order_index
  local arm=$1 slot=$2 order_i=$3
  local cli daemon out_dir home rc
  if [[ "$arm" == "baseline" ]]; then
    cli="$BASE_HIPFIRE"
    daemon="$BASE_DAEMON"
  else
    cli="$CAND_HIPFIRE"
    daemon="$CAND_DAEMON"
  fi
  out_dir="$RUN_DIR/product/${arm}-${slot}"
  mkdir -p "$out_dir"
  progress "PRODUCT START arm=$arm slot=$slot order=$order_i cli=$cli daemon=$daemon"

  home="$out_dir/home"
  mkdir -p "$home/.hipfire"

  {
    echo "arm=$arm"
    echo "slot=$slot"
    echo "order_index=$order_i"
    echo "cli=$cli"
    echo "daemon=$daemon"
    echo "cli_md5=$(md5sum "$cli" | awk '{print $1}')"
    echo "daemon_md5=$(md5sum "$daemon" | awk '{print $1}')"
    echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'argv='
    printf ' %q' "$cli" bench "${PRODUCT_MODEL_TAG:-$PRODUCT_MODEL}" \
      --spec dflash --runs 5 --warmups 3 --max-tokens 256 \
      --backend noslots --workload stateless --kv-mode q8 \
      --json --prompt-file "$PROMPT"
    printf '\n'
    echo "HIPFIRE_DAEMON_BIN=$daemon"
    echo "HIPFIRE_DFLASH_DRAFT=$PRODUCT_DRAFT"
    echo "HIPFIRE_LOCAL=1"
    echo "HIPFIRE_MODELS_DIR=${MODELS_DIR:-/home/kaden/.hipfire/models}"
    echo "model=${PRODUCT_MODEL_TAG:-$PRODUCT_MODEL}"
  } > "$out_dir/command.txt"

  # Baseline CLI → baseline daemon; candidate CLI → candidate daemon.
  set +e
  HIPFIRE_HOME="$home" \
  HIPFIRE_MODELS_DIR="${MODELS_DIR:-/home/kaden/.hipfire/models}" \
  HIPFIRE_LOCAL=1 \
  HIPFIRE_DAEMON_BIN="$daemon" \
  HIPFIRE_VERIFY_GRAPH=0 \
  HIPFIRE_NO_REGISTRY_FETCH=1 \
  HIPFIRE_DFLASH_DRAFT="$PRODUCT_DRAFT" \
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  env -u HIPFIRE_CLI_BIN -u HIPFIRE_KERNEL_CACHE \
    "$cli" bench "${PRODUCT_MODEL_TAG:-$PRODUCT_MODEL}" \
      --spec dflash \
      --runs 5 \
      --warmups 3 \
      --max-tokens 256 \
      --backend noslots \
      --workload stateless \
      --kv-mode q8 \
      --json \
      --prompt-file "$PROMPT" \
    >"$out_dir/stdout" 2>"$out_dir/stderr"
  rc=$?
  set -e
  echo "$rc" > "$out_dir/exit_status"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out_dir/end.iso"

  # Extract JSON report from stdout if present (pretty-printed object).
  if python3 - "$out_dir/stdout" "$out_dir/report.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, "r", encoding="utf-8", errors="replace").read()
decoder = json.JSONDecoder()
last = None
i = 0
while i < len(text):
    if text[i] == "{":
        try:
            obj, end = decoder.raw_decode(text, i)
            last = obj
            i = end
            continue
        except json.JSONDecodeError:
            pass
    i += 1
if last is None:
    open(dst, "w", encoding="utf-8").write("{}\n")
    raise SystemExit(2)
json.dump(last, open(dst, "w", encoding="utf-8"), indent=2, sort_keys=True)
open(dst, "a", encoding="utf-8").write("\n")
PY
  then
    echo "report=present" > "$out_dir/report_status.txt"
  else
    echo "report=missing_or_unparsed" > "$out_dir/report_status.txt"
  fi

  {
    grep -E 'decode_tok_s|tok/s|mean|median|prompt_md5|prompt_tokens' "$out_dir/stdout" 2>/dev/null || true
    grep -E 'decode_tok_s|tok/s|mean|median|error|panic' "$out_dir/stderr" 2>/dev/null || true
  } > "$out_dir/metrics_scrape.txt" || true

  python3 - "$out_dir/report.json" "$out_dir/metrics.txt" <<'PY' || true
import json, sys
rep_path, out_path = sys.argv[1], sys.argv[2]
try:
    rep = json.load(open(rep_path, encoding="utf-8"))
except Exception as e:
    open(out_path, "w", encoding="utf-8").write(f"parse_error={e}\n")
    raise SystemExit(0)

def dig(d, *ks):
    cur = d
    for k in ks:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

lines = []
for key in ("decode_tok_s", "prefill_tok_s", "wall_tok_s", "ttft_ms"):
    stats = dig(rep, key)
    if isinstance(stats, dict):
        for sk in ("mean", "median", "min", "max", "std"):
            if sk in stats:
                lines.append(f"{key}.{sk}={stats[sk]}")
    elif stats is not None:
        lines.append(f"{key}={stats}")
for key in ("prompt_tokens", "prompt_md5", "prompt_chars", "max_tokens", "runs"):
    if key in rep:
        lines.append(f"{key}={rep[key]}")
samples = dig(rep, "samples", "decode")
if isinstance(samples, list):
    lines.append(f"decode_samples={json.dumps(samples)}")
open(out_path, "w", encoding="utf-8").write("\n".join(lines) + ("\n" if lines else ""))
PY

  {
    echo "rc=$rc"
    cat "$out_dir/metrics.txt" 2>/dev/null || true
    cat "$out_dir/report_status.txt"
  } > "$out_dir/status.txt"

  progress "PRODUCT DONE arm=$arm slot=$slot rc=$rc $(tr '\n' ' ' < "$out_dir/metrics.txt" 2>/dev/null | cut -c1-160)"
  if [[ "$rc" -ne 0 ]]; then
    echo "[packed-ab] PRODUCT STDERR TAIL:" >&2
    tail -60 "$out_dir/stderr" >&2 || true
    die "product arm=$arm slot=$slot exited rc=$rc (no autofallback)"
  fi
}

# ---------------------------------------------------------------------------
# Route 1: demo ABBAABBA (A=baseline, B=candidate) — 4 each, fresh process
# ---------------------------------------------------------------------------
progress "ROUTE demo START order=ABBAABBA"
DEMO_ORDER=(baseline candidate candidate baseline baseline candidate candidate baseline)
declare -A DEMO_SLOT=([baseline]=0 [candidate]=0)
order_i=0
for arm in "${DEMO_ORDER[@]}"; do
  slot=${DEMO_SLOT[$arm]}
  run_demo_once "$arm" "$slot" "$order_i"
  DEMO_SLOT[$arm]=$((slot + 1))
  order_i=$((order_i + 1))
done
progress "ROUTE demo DONE baseline_slots=${DEMO_SLOT[baseline]} candidate_slots=${DEMO_SLOT[candidate]}"

# ---------------------------------------------------------------------------
# Route 2: product ABBAAB (3 each), fresh process per arm
# ---------------------------------------------------------------------------
progress "ROUTE product START order=ABBAAB"
PRODUCT_ORDER=(baseline candidate candidate baseline baseline candidate)
declare -A PRODUCT_SLOT=([baseline]=0 [candidate]=0)
order_i=0
for arm in "${PRODUCT_ORDER[@]}"; do
  slot=${PRODUCT_SLOT[$arm]}
  run_product_once "$arm" "$slot" "$order_i"
  PRODUCT_SLOT[$arm]=$((slot + 1))
  order_i=$((order_i + 1))
done
progress "ROUTE product DONE baseline_slots=${PRODUCT_SLOT[baseline]} candidate_slots=${PRODUCT_SLOT[candidate]}"

# ---------------------------------------------------------------------------
# JSON summary (routes isolated; never mixes demo rates into product)
# ---------------------------------------------------------------------------
python3 - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])

def read_kv(path: Path):
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def collect_route(route: str):
    root = run_dir / route
    runs = []
    if not root.is_dir():
        return runs
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if "-" not in name:
            continue
        arm, slot_s = name.rsplit("-", 1)
        try:
            slot = int(slot_s)
        except ValueError:
            continue
        st = read_kv(d / "status.txt")
        metrics = read_kv(d / "metrics.txt")
        rc_s = (d / "exit_status").read_text().strip() if (d / "exit_status").is_file() else st.get("rc")
        try:
            rc = int(rc_s) if rc_s is not None else None
        except ValueError:
            rc = None
        entry = {
            "arm": arm,
            "slot": slot,
            "dir": str(d),
            "exit_status": rc,
            "stdout": str(d / "stdout"),
            "stderr": str(d / "stderr"),
            "command": str(d / "command.txt"),
            "metrics": metrics,
            "status": st,
        }
        if route == "product":
            entry["report_json"] = str(d / "report.json")
            entry["report"] = load_json(d / "report.json")
        runs.append(entry)
    runs.sort(key=lambda r: (0 if r["arm"] == "baseline" else 1, r["slot"]))
    return runs

def arm_stats(runs, arm, key):
    vals = []
    for r in runs:
        if r["arm"] != arm:
            continue
        m = r.get("metrics") or {}
        raw = m.get(key)
        if raw is None or raw == "":
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = sum(vals_sorted) / n
    med = vals_sorted[n // 2] if n % 2 else 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
    return {"n": n, "mean": mean, "median": med, "min": vals_sorted[0], "max": vals_sorted[-1], "values": vals_sorted}

demo_runs = collect_route("demo")
product_runs = collect_route("product")
bins = load_json(run_dir / "meta" / "bins.json") or {}

labeled_md5 = {}
lp = run_dir / "hashes" / "labeled.md5"
if lp.is_file():
    for line in lp.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            labeled_md5[parts[0]] = parts[1]

summary = {
    "schema": "packed-ab-v1",
    "run_dir": str(run_dir),
    "git_head": bins.get("git_head"),
    "prompt_md5": bins.get("prompt_md5"),
    "binary_md5": labeled_md5,
    "bins": bins,
    "note": "Routes are independent. NEVER compare demo tok/s against product bench tok/s.",
    "demo": {
        "order": "ABBAABBA",
        "runs_per_arm": 4,
        "runs": demo_runs,
        "baseline_decode_tok_s": arm_stats(demo_runs, "baseline", "decode_tok_s"),
        "candidate_decode_tok_s": arm_stats(demo_runs, "candidate", "decode_tok_s"),
        "baseline_decode_tau": arm_stats(demo_runs, "baseline", "decode_tau"),
        "candidate_decode_tau": arm_stats(demo_runs, "candidate", "decode_tau"),
        "token_sha8_by_run": [
            {"arm": r["arm"], "slot": r["slot"], "token_sha8": (r.get("metrics") or {}).get("token_sha8")}
            for r in demo_runs
        ],
    },
    "product": {
        "order": "ABBAAB",
        "runs_per_arm": 3,
        "runs": product_runs,
        "baseline_decode_tok_s": arm_stats(product_runs, "baseline", "decode_tok_s.mean")
            or arm_stats(product_runs, "baseline", "decode_tok_s"),
        "candidate_decode_tok_s": arm_stats(product_runs, "candidate", "decode_tok_s.mean")
            or arm_stats(product_runs, "candidate", "decode_tok_s"),
    },
}

for route_key in ("demo", "product"):
    route = summary[route_key]
    b = route.get("baseline_decode_tok_s") or {}
    c = route.get("candidate_decode_tok_s") or {}
    bm, cm = b.get("mean"), c.get("mean")
    if isinstance(bm, (int, float)) and isinstance(cm, (int, float)) and bm > 0:
        route["within_route_gain_pct"] = (cm - bm) / bm * 100.0
    else:
        route["within_route_gain_pct"] = None

out = run_dir / "summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[packed-ab] SUMMARY wrote {out}")
for route_key in ("demo", "product"):
    r = summary[route_key]
    bstat = r.get("baseline_decode_tok_s")
    cstat = r.get("candidate_decode_tok_s")
    gain = r.get("within_route_gain_pct")
    print(f"[packed-ab] {route_key}: baseline={bstat} candidate={cstat} gain_pct={gain}")
PY

exec 9>&-
gpu_release
echo "[packed-ab] ALL-DONE run_dir=$RUN_DIR summary=$RUN_DIR/summary.json"
