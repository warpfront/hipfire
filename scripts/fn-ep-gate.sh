#!/usr/bin/env bash
# fn-ep-gate.sh: Flash-Next (Qwen4) expert-parallel gate for a host with N homogeneous
# gfx1100 (7900 XTX) or gfx1201 (R9700) cards. Run it from a hipfire checkout; it builds the
# binaries, runs a reference (one card by default) and N cards (topology.ep = N) over the same prompts,
# compares them byte for byte, times decode/prefill, and writes a receipt you can send back.
# With --mode pp the same gate qualifies Flash-Next pipeline parallelism (pp = N layer bands, one per card, HC
# streams over host-mapped staging, the output card last): a reference and PP-N run the same prompts through
# fn_ep_oracle --pp N --canonical-state and are compared byte for byte (see "PP mode" below).
#
#   scripts/fn-ep-gate.sh [options] MODEL [N] [OUT_DIR]
#
#   MODEL        Flash-Next model file (REQUIRED). EP mode: the mq4-xts file. --mode pp: the packed mq4 / mq4-xts file or an
#                expert-slab container (`*.mq3-xts`, `*.mq2-xts`). Slab files load under PP on the cards their codecs support
#                (gfx1201: mq3-xts and mq2-xts; gfx1100: mq2-xts only). The script does not look at the file name: it
#                passes the path through and hashes only that file (the PLE row files beside a split base are not hashed).
#   N            EP degree (--mode pp: pipeline degree), 2..=8 (default: length of --devices, else every visible card of the chosen arch, at most 8)
#   OUT_DIR      output directory (default: ./fn-ep-gate-<UTC timestamp>)
#
# Run the arms inside whole-card fabric holds: automatic expert placement can fill each card.
# On 32 GB R9700s the EP2 reference / EP4 target measured 31.17 / 30.68 GiB per-card peaks.
#
# Options (also accepted as --n N / --out DIR before or after the positionals):
#   --arch ARCH         card arch to use (default: gfx1100 or gfx1201, whichever has more cards)
#   --devices LIST      comma list of cards in any hardware.devices form (PCI-order index, GPU-<uuid>, PCI
#                       address, gfxNNNN). HIPFIRE_DEVICES for the EP arm is LIST verbatim, the single arm
#                       uses its first entry, N defaults to its length. Default: an inherited HIPFIRE_DEVICES
#                       (e.g. a scheduler's held cards) when set, else the first N cards of --arch from KFD.
#                       ROCR_VISIBLE_DEVICES is cleared in the arms only for the KFD-selected default.
#   --tokens T          greedy tokens per prompt (default 128)
#   --max-seq S         KV context (default 8192)
#   --state-steps L     state digest steps (default 1,64,128)
#   --reps R            fresh processes per speed cell and arm (default 3)
#   --skip-build        reuse existing binaries (cargo target dir: $CARGO_TARGET_DIR or ./target)
#   --skip-speed        exactness only (no `hipfire bench` cells)
#   --skip-model-sha    do not hash the model file (the receipt then has no model sha256)
#   --ref-ranks R       reference arm EP degree, 1 <= R < N (default 1 = one card). Use it when one card
#                       would need more pinned host RAM than you have (a 24 GB card pins ~50 GiB of
#                       experts; EP4 on 24 GB cards keeps every expert layer in VRAM). The reference then
#                       runs EP-R on the first R cards (HIPFIRE_DEVICES = first R entries of the card
#                       list) as arm `ref-epR`, and the EP-N arm is compared against it byte for byte.
#   --mode ep|pp|reserve  gate to run (default ep). pp = Flash-Next pipeline parallelism, see "PP mode" below; reserve =
#                       the opt-in EP2 reserve-decoupling qualification, see "Reserve mode" below.
#   --pp-layers COUNTS  (pp) N comma-separated layer counts, band 0 first, exported as HIPFIRE_PP_LAYERS to the PP-N
#                       arm (default: the uniform split; the PLE layer is pinned to band 0). A PP-R reference
#                       (--ref-ranks R > 1) always runs the uniform split: independent bands.
#   --transport auto|host  (pp) HIPFIRE_MESH_TRANSPORT of the PP arms (default host; auto resolves to host-mapped
#                       staging; peer edges are refused by the loader)
#   --mtp               (pp) add the native-MTP exactness pair (`--spec mtp --temp 0 --ngram off` with the pinned
#                       policy HIPFIRE_MTP_AR_FLOOR=0 HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_NGRAM=off); the AR pair
#                       always runs
#   --no-isolate        keep your own HOME / HIPFIRE_HOME (default: a private one per arm under OUT_DIR
#                       holding only `topology.ep`, so your config cannot change the arms)
#   --target-dir DIR    cargo target dir
#   -h, --help          this text
#
# Environment: EXTRA_ENV="K=V K2=V2" adds variables to every arm; FN_EP_GATE_ARM_TIMEOUT (7200 s,
# oracle arm) and FN_EP_GATE_BENCH_TIMEOUT (3600 s, one bench process) bound each run.
#
# Steps: (a) environment receipt (git/build hash, ROCm, rocm-smi topology + P2P access, per-card
# arch/VRAM/PCIe link speed+width from sysfs, host RAM); (b) cargo build of fn_ep_oracle
# (hipfire-loader, feature `lab`), hipfire, daemon; (c) arms `single` (HIPFIRE_DEVICES = first card,
# HIPFIRE_TOPOLOGY_EP=1) and `epN` (first N cards, HIPFIRE_TOPOLOGY_EP=N) through fn_ep_oracle with
# the two prompts in scripts/fn-ep-gate-prompts (~500 and ~4,400 tokens), 128 greedy tokens, state
# digests at steps 1,64,128 (with --ref-ranks R > 1 the first arm is `ref-epR`: the first R cards,
# HIPFIRE_TOPOLOGY_EP=R, and every `single` below reads `ref-epR`); (d) scripts/fn_ep_compare.py
# (tokens, rank-0 logits sha, every peer logits mirror sha, every rank's state digests: all must
# equal the single card byte for byte; an EP-R reference is also checked against itself);
# (e) speed cells, `hipfire bench --spec off` decode (500-token prompt, 128 tokens) and prefill
# (4,400-token prompt, 1 token), REPS fresh processes per arm, A/B interleaved, raw JSON kept;
# (f) OUT_DIR/receipt.json and OUT_DIR/summary.txt.
#
# A failed EP load (missing peer access, out of VRAM, ...) is recorded in the receipt with the
# stderr tail; the script never exits without writing receipt.json. Exit status: 0 = byte-exact PASS,
# 1 = mismatch or an arm failed, 2 = environment / usage problem (not enough cards, build failed).
#
# PP mode (--mode pp): N is the pipeline degree (2..=8). The first entry of HIPFIRE_DEVICES / --devices is the
# output card and runs the LAST band (ranks 0..N-1 are the list entries 2..N, then entry 1). Arms, all with
# topology.ep = 1 and --canonical-state: `single` (--ref-ranks 1: one card, pp = 1, experts host-mapped by default)
# or `ref-ppR` (PP-R on the first R cards, uniform bands, HIPFIRE_TOPOLOGY_PP=R, --pp R) as the reference, and
# `ppN` (HIPFIRE_TOPOLOGY_PP=N, --pp N, HIPFIRE_PP_LAYERS, HIPFIRE_MESH_TRANSPORT). Any cut-safe band split is bitwise
# equal to one uncut card, so PP-R is a valid reference when one card would need more pinned host RAM than the host
# has. With --mtp the arms `single-mtp` (`ref-ppR-mtp`) and `ppN-mtp` also run. The arms clear inherited
# HIPFIRE_PP_LAYERS / HIPFIRE_MESH_TRANSPORT / HIPFIRE_TOPOLOGY_PP / HIPFIRE_MTP_* and the expert-placement variables
# the loader refuses under pp (HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS, _EXPERT_HOTSET, _EXPERT_STAGE, _TURN_SNAPSHOTS).
# Comparison: OUT_DIR/pp_compare.py (it reuses the helpers of scripts/fn_ep_compare.py, which itself needs EP rank
# mirrors). Only when both metas carry canonical_state and ranks == pp are `ranks` not compared; then meta
# (completed, tokens, max_seq, spec, steps_run, prompts, state_steps, vocab), every logits/window/tokens/state
# record and every raw logits dump must be identical, and each arm's meta.json must record the requested pp and
# placement (band counts, output-last rank order matching the card list, host-mapped edges, pinned env).
# Speed: the same `hipfire bench` cells with HIPFIRE_TOPOLOGY_PP=pp and topology.pp = pp in the arm's private
# home (the bench's own --pp is its prompt-size list and is not used). Every cell's bench JSON must carry numeric
# `pp` and `loaded.pp` equal to the arm's pipeline degree (1 for the single-card reference), or the cell fails
# and the gate cannot PASS. Receipt: the EP structure plus mode, pp, bands, transport,
# per-arm placement metadata, the ranks mapped output-last (PCI id, arch, link speed/width) and host RAM.
#
# Reserve mode (--mode reserve): the opt-in EP2 reserve-decoupling qualification (HIPFIRE_EP_RESERVE_DECOUPLE, see
# docs/multi-gpu.md "EP reserve decoupling"). gfx1201 EP2 only (N defaults to 2, --arch to gfx1201, --max-seq to 16384).
# One binary, two same-schedule arms with identical devices, model and prompts: `ep2-off` (switch unset) and `ep2-on`
# (HIPFIRE_EP_RESERVE_DECOUPLE=1); --mtp adds `ep2-off-mtp` / `ep2-on-mtp` (native MTP, greedy, pinned policy as in pp
# mode); --baseline-repeat adds `ep2-off-repeat` (off vs off repeatability). Every reserve arm clears
# HIPFIRE_PREFILL_CHUNK_ROWS (the prompt chunk stays 4096, never lowered) and EXTRA_ENV may not set either variable.
# EVERY reserve arm (off arms ep2-off, ep2-off-mtp, ep2-off-repeat included) sets HIPFIRE_EP_RESERVE_RECEIPT=1, a developer env
# that only prints host-side `EP_RESERVE_RECEIPT <json>` lines (the on arms add the donor-hash read-back): the oracle's header,
# ids and end records, and from the forward the per-rank header/rank records (the real resident expert layer count and prompt
# chunk of an ordinary load) and one `prompt_rank` record per rank per outermost prompt entry, keyed (prompt, phase,
# ordinal). `fn_ep_compare.py reserve harvest` merges the two ranks' records per key; the off arm's receipts come from the real
# ordinary load and forward, never from the expected plan.
#   --boundary-dir DIR     REQUIRED: *.txt / *.ids prompt files of exact token counts (*.ids = whitespace-separated token ids;
#                          sorted, passed without truncation); the
#                          oracle's reported token counts must cover --boundary-lengths
#   --boundary-lengths L   comma list the corpus must cover (default 8,9,511,512,513,2047,2048,4095,4096,4097,4396)
#   --min-resident N       resident expert layers required on every rank with the switch on (default 33; fewer = kill)
#   --seed S               oracle --seed, recorded in the receipts (default 1; the arms are greedy)
#   --baseline-repeat      also run `ep2-off-repeat` and compare it to `ep2-off`
#   --perf-json FILE       (repeatable) ep-reserve-perf/1 document from the card campaign: the pp32K kill rule (throughput
#                          ratio < 0.95 or restore-inclusive latency ratio > 1.05); other cells are reported, not judged
#   --c1-manifest FILE     ep-reserve-c1/1 manifest: the weighted sampled C1 kill rule (gain < +8%)
# Every arm must print `EP_RESERVE_RECEIPT <json>` lines (kinds header/rank/prompt_rank/ids/end) or leave reserve-receipt.json
# in its oracle output directory; `fn_ep_compare.py reserve harvest` assembles them. The pair check requires both rank
# receipts, plan/source/seed agreement, 4096-row chunking, >= --min-resident layers, restore timing, donor hashes, a stable
# exchange address and equal committed id arrays, on top of the byte-exact logits/mirror/state comparison. Without
# --perf-json / --c1-manifest only Gate 1 is evaluated and summary.txt says so. Nothing in this mode is a performance
# claim; the thresholds are kill rules. Outputs: compare-<gate>.txt/.json, perf-*.json, c1.json, receipt.json (reserve).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PROMPT_DIR=$SCRIPT_DIR/fn-ep-gate-prompts
COMPARE=$SCRIPT_DIR/fn_ep_compare.py
SHORT_PROMPT=$PROMPT_DIR/short-500.txt
LONG_PROMPT=$PROMPT_DIR/long-4400.txt
LONG_PROMPT_TOKENS=4400
PROMPT_ARGS=(--prompt-file "$SHORT_PROMPT" --prompt-file "$LONG_PROMPT" --truncate-prompt-tokens "$LONG_PROMPT_TOKENS")

MODEL="" N="" OUT="" ARCH=""
REF_RANKS=1
MODE=ep PP_LAYERS="" TRANSPORT=host MTP=0 PP_OPTS=""
BOUNDARY_DIR="" BOUNDARY_LENGTHS=8,9,511,512,513,2047,2048,4095,4096,4097,4396 MIN_RESIDENT=33 SEED=1 BASELINE_REPEAT=0
PERF_JSONS=() C1_MANIFEST="" RESERVE_OPTS="" RESERVE_ARMS="" MAX_SEQ_SET=0 BOUNDARY_FILES=()
RESERVE_MIN_TOKENS=128 RES_ANY_FAIL=0 RES_ANY_ERR=0
TOKENS=128 MAX_SEQ=8192 STATE_STEPS="1,64,128" REPS=3
SKIP_BUILD=0 SKIP_SPEED=0 SKIP_SHA=0 NO_ISOLATE=0
ARM_TIMEOUT=${FN_EP_GATE_ARM_TIMEOUT:-7200}
BENCH_TIMEOUT=${FN_EP_GATE_BENCH_TIMEOUT:-3600}
EXTRA_ENV=${EXTRA_ENV:-}
DEVLIST="" DEVS_SOURCE=kfd

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "fn-ep-gate: $*" >&2; exit 2; }
log() { echo "[fn-ep-gate $(date -u +%H:%M:%S)] $*"; }

POS=()
while (($#)); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --n|-n) [[ $# -ge 2 ]] || die "$1 needs a value"; N=$2; shift 2 ;;
    --out) [[ $# -ge 2 ]] || die "$1 needs a value"; OUT=$2; shift 2 ;;
    --arch) [[ $# -ge 2 ]] || die "$1 needs a value"; ARCH=$2; shift 2 ;;
    --devices) [[ $# -ge 2 ]] || die "$1 needs a value"; DEVLIST=${2// /}; shift 2 ;;
    --tokens) [[ $# -ge 2 ]] || die "$1 needs a value"; TOKENS=$2; shift 2 ;;
    --max-seq) [[ $# -ge 2 ]] || die "$1 needs a value"; MAX_SEQ=$2; MAX_SEQ_SET=1; shift 2 ;;
    --state-steps) [[ $# -ge 2 ]] || die "$1 needs a value"; STATE_STEPS=$2; shift 2 ;;
    --reps) [[ $# -ge 2 ]] || die "$1 needs a value"; REPS=$2; shift 2 ;;
    --target-dir) [[ $# -ge 2 ]] || die "$1 needs a value"; export CARGO_TARGET_DIR=$2; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-speed) SKIP_SPEED=1; shift ;;
    --skip-model-sha) SKIP_SHA=1; shift ;;
    --ref-ranks) [[ $# -ge 2 ]] || die "$1 needs a value"; REF_RANKS=$2; shift 2 ;;
    --no-isolate) NO_ISOLATE=1; shift ;;
    --mode) [[ $# -ge 2 ]] || die "$1 needs a value"; MODE=$2; shift 2 ;;
    --pp-layers) [[ $# -ge 2 ]] || die "$1 needs a value"; PP_LAYERS=${2// /}; PP_OPTS+=" --pp-layers"; shift 2 ;;
    --transport) [[ $# -ge 2 ]] || die "$1 needs a value"; TRANSPORT=$2; PP_OPTS+=" --transport"; shift 2 ;;
    --mtp) MTP=1; PP_OPTS+=" --mtp"; shift ;;
    --boundary-dir) [[ $# -ge 2 ]] || die "$1 needs a value"; BOUNDARY_DIR=$2; RESERVE_OPTS+=" --boundary-dir"; shift 2 ;;
    --boundary-lengths) [[ $# -ge 2 ]] || die "$1 needs a value"; BOUNDARY_LENGTHS=${2// /}; RESERVE_OPTS+=" --boundary-lengths"; shift 2 ;;
    --min-resident) [[ $# -ge 2 ]] || die "$1 needs a value"; MIN_RESIDENT=$2; RESERVE_OPTS+=" --min-resident"; shift 2 ;;
    --seed) [[ $# -ge 2 ]] || die "$1 needs a value"; SEED=$2; RESERVE_OPTS+=" --seed"; shift 2 ;;
    --baseline-repeat) BASELINE_REPEAT=1; RESERVE_OPTS+=" --baseline-repeat"; shift ;;
    --perf-json) [[ $# -ge 2 ]] || die "$1 needs a value"; PERF_JSONS+=("$2"); RESERVE_OPTS+=" --perf-json"; shift 2 ;;
    --c1-manifest) [[ $# -ge 2 ]] || die "$1 needs a value"; C1_MANIFEST=$2; RESERVE_OPTS+=" --c1-manifest"; shift 2 ;;
    --) shift; POS+=("$@"); break ;;
    -*) die "unknown option $1 (try --help)" ;;
    *) POS+=("$1"); shift ;;
  esac
done
((${#POS[@]} <= 3)) || die "too many positional arguments"
[[ -z ${POS[0]:-} ]] || MODEL=${POS[0]}
[[ -z ${POS[1]:-} ]] || N=${POS[1]}
[[ -z ${POS[2]:-} ]] || OUT=${POS[2]}
[[ -n $MODEL ]] || { usage >&2; die "MODEL is required"; }
[[ -f $MODEL && -r $MODEL ]] || die "model file not readable: $MODEL"
[[ -z $N || $N =~ ^[0-9]+$ ]] || die "N must be an integer"
if [[ -z $DEVLIST && -n ${HIPFIRE_DEVICES:-} ]]; then
  DEVLIST=${HIPFIRE_DEVICES// /}
  DEVS_SOURCE=inherited
  echo "fn-ep-gate: using inherited HIPFIRE_DEVICES=$DEVLIST as the card list (ROCR_VISIBLE_DEVICES left as inherited)"
elif [[ -n $DEVLIST ]]; then
  DEVS_SOURCE=flag
fi
[[ $TOKENS =~ ^[0-9]+$ && $MAX_SEQ =~ ^[0-9]+$ && $REPS =~ ^[0-9]+$ ]] || die "--tokens/--max-seq/--reps must be integers"
[[ $REF_RANKS =~ ^[0-9]+$ ]] || die "--ref-ranks must be an integer"
[[ $REF_RANKS =~ ^0*([1-7])$ ]] || die "--ref-ranks must be in 1..7 and less than N"
REF_RANKS=${BASH_REMATCH[1]}
[[ $MODE == ep || $MODE == pp || $MODE == reserve ]] || die "--mode must be ep, pp or reserve"
if [[ $MODE == ep ]]; then
  [[ -z $PP_OPTS ]] || die "${PP_OPTS# } need --mode pp"
elif [[ $MODE == reserve ]]; then
  [[ -z ${PP_OPTS// --mtp/} ]] || die "${PP_OPTS# } need --mode pp (only --mtp applies to --mode reserve)"
else
  [[ $TRANSPORT == auto || $TRANSPORT == host ]] || die "--transport must be auto or host (peer edges are refused by the loader)"
  [[ -z $PP_LAYERS || $PP_LAYERS =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || die "--pp-layers must be comma-separated positive layer counts (for example 9,8,8,8,8,7)"
fi
if [[ $MODE != reserve ]]; then
  [[ -z $RESERVE_OPTS ]] || die "${RESERVE_OPTS# } need --mode reserve"
else
  [[ -z $N || $N == 2 ]] || die "--mode reserve is EP2 only (N must be 2)"
  N=2
  [[ -n $ARCH ]] || ARCH=gfx1201
  [[ $ARCH == gfx1201 ]] || die "--mode reserve qualifies gfx1201 EP2 only (--arch $ARCH); other archs and topologies keep today's ordinary path"
  [[ -n $BOUNDARY_DIR && -d $BOUNDARY_DIR ]] || die "--mode reserve needs --boundary-dir DIR (exact-length *.txt / *.ids prompt files covering --boundary-lengths)"
  BOUNDARY_DIR=$(readlink -f "$BOUNDARY_DIR")
  mapfile -t BOUNDARY_FILES < <(find "$BOUNDARY_DIR" -maxdepth 1 -type f \( -name '*.txt' -o -name '*.ids' \) | LC_ALL=C sort)
  ((${#BOUNDARY_FILES[@]} >= 1)) || die "no *.txt / *.ids prompt files in $BOUNDARY_DIR"
  [[ $BOUNDARY_LENGTHS =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || die "--boundary-lengths must be a comma list of positive token counts"
  [[ $MIN_RESIDENT =~ ^[1-9][0-9]*$ && $SEED =~ ^[0-9]+$ ]] || die "--min-resident / --seed must be integers"
  ((TOKENS >= RESERVE_MIN_TOKENS)) || die "--tokens must be >= $RESERVE_MIN_TOKENS in reserve mode (committed-token requirement)"
  [[ $EXTRA_ENV != *HIPFIRE_EP_RESERVE_DECOUPLE* && $EXTRA_ENV != *HIPFIRE_PREFILL_CHUNK_ROWS* ]] || die "EXTRA_ENV may not set HIPFIRE_EP_RESERVE_DECOUPLE or HIPFIRE_PREFILL_CHUNK_ROWS in reserve mode"
  ((MAX_SEQ_SET)) || MAX_SEQ=16384
  _abs=()
  for _f in ${PERF_JSONS[@]+"${PERF_JSONS[@]}"}; do
    [[ -r $_f ]] || die "--perf-json not readable: $_f"
    _abs+=("$(readlink -f "$_f")")
  done
  PERF_JSONS=(${_abs[@]+"${_abs[@]}"})
  [[ -z $C1_MANIFEST ]] || { [[ -r $C1_MANIFEST ]] || die "--c1-manifest not readable: $C1_MANIFEST"; C1_MANIFEST=$(readlink -f "$C1_MANIFEST"); }
fi
((REPS >= 1)) || die "--reps must be >= 1"
[[ -f $SHORT_PROMPT && -f $LONG_PROMPT && -f $COMPARE ]] || die "scripts/fn-ep-gate-prompts or fn_ep_compare.py missing next to this script"
command -v python3 >/dev/null || die "python3 is required"
command -v cargo >/dev/null || [[ $SKIP_BUILD == 1 ]] || die "cargo is required (or pass --skip-build)"

MODEL=$(readlink -f "$MODEL")
[[ -n $OUT ]] || OUT=./fn-ep-gate-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
mkdir -p "$OUT/env" "$OUT/oracle" "$OUT/speed"
exec > >(tee -a "$OUT/gate.log") 2>&1

TARGET_DIR=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
ORACLE=$TARGET_DIR/release/examples/fn_ep_oracle
HIPFIRE=$TARGET_DIR/release/hipfire
DAEMON=$TARGET_DIR/release/daemon

DEVS="" BDFS=() SINGLE_DEVS="" CARD_IDX=""
REF_ARM=single REF_DEVS=""
SAMPLER=""
VERDICT="" REASON=""
BUILD_RC="" ORACLE_RC_SINGLE="" ORACLE_RC_EP="" COMPARE_STATE="not run"
FINAL_RC=""
STARTED=$(date -u +%FT%TZ)
RECEIPT_DONE=0
MTP_STATE="not run" SPEED_FAIL=0 PP_ANY_ERR=0 PP_ANY_FAIL=0 PAIR_RC=0 PAIR_STATE=""
PP_PINNED_MTP=(HIPFIRE_MTP_AR_FLOOR=0 HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_NGRAM=off)

# ------------------------------------------------------------------ receipt
write_receipt() {
  ((RECEIPT_DONE == 0)) || return 0
  RECEIPT_DONE=1
  export GATE_OUT=$OUT GATE_N=$N GATE_ARCH=$ARCH GATE_DEVS=$DEVS GATE_CARD_IDX=$CARD_IDX GATE_DEVS_SOURCE=$DEVS_SOURCE GATE_MODEL=$MODEL GATE_VERDICT=$VERDICT \
    GATE_REASON=$REASON GATE_STARTED=$STARTED GATE_REPS=$REPS GATE_TOKENS=$TOKENS GATE_MAX_SEQ=$MAX_SEQ \
    GATE_STATE_STEPS=$STATE_STEPS GATE_BUILD_RC=$BUILD_RC GATE_COMPARE_STATE=$COMPARE_STATE \
    GATE_SKIP_SPEED=$SKIP_SPEED GATE_NO_ISOLATE=$NO_ISOLATE GATE_EXTRA_ENV=$EXTRA_ENV \
    GATE_LONG_TOKENS=$LONG_PROMPT_TOKENS GATE_REF_RANKS=$REF_RANKS GATE_REF_ARM=$REF_ARM \
    GATE_MODE=$MODE GATE_TARGET_ARM=$MODE$N GATE_PP_LAYERS=$PP_LAYERS GATE_TRANSPORT=$TRANSPORT GATE_MTP=$MTP \
    GATE_MTP_STATE=$MTP_STATE GATE_SPEED_FAIL=$SPEED_FAIL \
    GATE_RESERVE_ARMS=$RESERVE_ARMS GATE_MIN_RESIDENT=$MIN_RESIDENT
  python3 - <<'PY' || echo "fn-ep-gate: WARNING: receipt writer failed" >&2
import json, os, re, statistics, datetime

E = os.environ
out = E["GATE_OUT"]
n = int(E["GATE_N"]) if E.get("GATE_N", "").isdigit() else None
ref_ranks = int(E["GATE_REF_RANKS"]) if E.get("GATE_REF_RANKS", "").isdigit() else 1
ref_arm = E.get("GATE_REF_ARM") or "single"
mode = E.get("GATE_MODE") or "ep"
mtp = E.get("GATE_MTP") == "1"
target_arm = "%s%d" % (mode, n) if n else mode
arm_names = [ref_arm, target_arm] + ([ref_arm + "-mtp", target_arm + "-mtp"] if mtp else [])
if mode == "reserve":
    arm_names = E.get("GATE_RESERVE_ARMS", "").split()


def rd(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def rdj(path):
    text = rd(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        start = text.find("{")
        if start >= 0:
            try:
                return json.loads(text[start:])
            except ValueError:
                return None
        return None


def tail(path, count=40):
    text = rd(path)
    return None if text is None else "\n".join(text.splitlines()[-count:])


def seconds(rec):
    try:
        a = datetime.datetime.strptime(rd(rec + "/start.txt").strip(), "%Y-%m-%dT%H:%M:%SZ")
        b = datetime.datetime.strptime(rd(rec + "/end.txt").strip(), "%Y-%m-%dT%H:%M:%SZ")
        return int((b - a).total_seconds())
    except Exception:
        return None


def rc_of(path):
    text = rd(path)
    try:
        return int(text.strip())
    except Exception:
        return None


FAILURE_PATTERNS = [
    ("peer_access", r"peer access|peer-access|canAccessPeer|EnablePeerAccess"),
    ("out_of_memory", r"out of memory|hipErrorOutOfMemory|\bOOM\b|insufficient (?:vram|memory)|does not fit"),
    ("arch_refused", r"requires gfx\d+|not supported on gfx\d+"),
    ("devices", r"HIPFIRE_DEVICES|hardware\.devices|visible devices"),
]


def classify(rc, err):
    if rc == 124:
        return "timeout"
    for name, pattern in FAILURE_PATTERNS:
        if err and re.search(pattern, err, re.I | re.M):
            return name
    return "error"


def peaks(path, bdf_count):
    text = rd(path)
    if not text:
        return None
    best = None
    for line in text.splitlines():
        parts = line.split()[1:]
        if not parts:
            continue
        vals = []
        for p in parts:
            try:
                vals.append(int(p))
            except ValueError:
                vals.append(0)
        best = vals if best is None else [max(a, b) for a, b in zip(best, vals)]
    return None if best is None else [round(v / (1 << 20)) for v in best]


cards = (rdj(out + "/env/cards.json") or {}).get("cards", [])
devs = [int(x) for x in E.get("GATE_CARD_IDX", "").split(",") if x.strip().isdigit()]
by_index = {c["index"]: c for c in cards}
chosen = [by_index[i] for i in devs if i in by_index]

env = {
    "host": rd(out + "/env/host.txt"),
    "cpu": rd(out + "/env/cpu.txt"),
    "meminfo": rd(out + "/env/meminfo.txt"),
    "cmdline": rd(out + "/env/cmdline.txt"),
    "git": rd(out + "/env/git.txt"),
    "hipfire_version": rd(out + "/env/hipfire-version.txt"),
    "rocm_version": rd(out + "/env/rocm-version.txt"),
    "hipcc_version": rd(out + "/env/hipcc-version.txt"),
    "driver": rd(out + "/env/driver.txt"),
    "rocm_smi_topology_and_access": rd(out + "/env/rocm-smi-topo.txt"),
    "inherited_env": rd(out + "/env/inherited-env.txt"),
    "model": rd(out + "/env/model.txt"),
    "cards_all": cards,
}
ordered = (chosen[1:] + chosen[:1]) if mode == "pp" else chosen
link_table = []
for rank, c in enumerate(ordered):
    row = {
        "rank": rank,
        "index": c["index"],
        "bdf": c["bdf"],
        "arch": c["arch"],
        "vram_total_mib": (c.get("vram_total_bytes") or 0) // (1 << 20),
        "link_speed": c.get("current_link_speed"),
        "link_width": c.get("current_link_width"),
        "max_link_speed": c.get("max_link_speed"),
        "max_link_width": c.get("max_link_width"),
        "pci_chain": c.get("pci_chain"),
    }
    if mode == "pp":
        # PP rank r is HIPFIRE_DEVICES entry (r + 1) mod N: the output card (entry 0) is the last rank.
        row["hipfire_device_position"] = chosen.index(c)
        row["role"] = "output card (last band)" if rank == len(ordered) - 1 else "band %d" % rank
    link_table.append(row)
p2p = None
if chosen:
    p2p = [
        [
            (next((l for l in a.get("p2p_links", []) if l.get("to_node") == b["node"]), None) if a is not b else "self")
            for b in chosen
        ]
        for a in chosen
    ]

arms = {}
for arm in arm_names:
    rec = "%s/oracle/%s-receipt" % (out, arm)
    if not os.path.isdir(rec):
        continue
    rc = rc_of(rec + "/rc.txt")
    err = tail(rec + "/stderr.txt", 40)
    meta = rdj("%s/oracle/%s/meta.json" % (out, arm))
    arms[arm] = {
        "rc": rc,
        "seconds": seconds(rec),
        "command": rd(rec + "/command.txt"),
        "completed": bool(meta and meta.get("completed")),
        "ranks": meta and meta.get("ranks"),
        "devices": meta and meta.get("devices"),
        "prompts": meta and meta.get("prompts"),
        "vram_peak_mib_per_card": peaks(rec + "/vram.txt", None),
        "failure": None if rc == 0 else {"kind": classify(rc, err), "stderr_tail": err},
    }
    if mode == "pp":
        arms[arm].update({
            "pp": meta.get("pp", 1) if meta else None,
            "canonical_state": meta and meta.get("canonical_state"),
            "band_counts": meta and meta.get("band_counts"),
            "rank_order": meta and meta.get("rank_order"),
            "edge_transports": meta and meta.get("edge_transports"),
            "output_owner": meta and meta.get("output_owner"),
            "numerical_policy": meta and meta.get("numerical_policy"),
            "spec": meta and meta.get("spec"),
            "meta": meta,
        })

compare = {"state": E.get("GATE_COMPARE_STATE")}
cj = rdj(out + "/compare.json")
if cj:
    compare.update({"verdict": cj.get("verdict"), "summary": cj.get("summary"), "first_mismatch": cj.get("first_mismatch"),
                    "families": cj.get("families"), "json": "compare.json", "text": "compare.txt"})
compare_mtp = None
if mtp:
    compare_mtp = {"state": E.get("GATE_MTP_STATE")}
    cmj = rdj(out + "/compare-mtp.json")
    if cmj:
        compare_mtp.update({"verdict": cmj.get("verdict"), "summary": cmj.get("summary"), "first_mismatch": cmj.get("first_mismatch"),
                            "families": cmj.get("families"), "json": "compare-mtp.json", "text": "compare-mtp.txt"})

speed = {}
sdir = out + "/speed"
if os.path.isdir(sdir):
    for arm in sorted(os.listdir(sdir)):
        cells = {}
        for cell in ("decode", "prefill"):
            runs = []
            for rep in range(1, int(E.get("GATE_REPS", "3")) + 1):
                base = "%s/%s/%s-%d" % (sdir, arm, cell, rep)
                if not os.path.exists(base + ".rc"):
                    continue
                rc = rc_of(base + ".rc")
                doc = rdj(base + ".json")
                samples = (doc or {}).get("samples", {})
                runs.append({
                    "rep": rep,
                    "rc": rc,
                    "decode_tok_s": (samples.get("decode") or [None])[0],
                    "prefill_tok_s": (samples.get("prefill") or [None])[0],
                    "prompt_tokens": (doc or {}).get("prompt_tokens"),
                    "raw": "speed/%s/%s-%d.json" % (arm, cell, rep),
                    "stderr_tail": None if rc == 0 else tail(base + ".stderr", 25),
                })
                if mode == "pp":
                    runs[-1]["effective_pp"] = (rd(base + ".pp") or "").strip() or None
            key = "decode_tok_s" if cell == "decode" else "prefill_tok_s"
            vals = [r[key] for r in runs if r["rc"] == 0 and r[key] is not None]
            cells[cell] = {"runs": runs, "metric": key, "median": statistics.median(vals) if vals else None, "n_ok": len(vals)}
        speed[arm] = cells
        skip = rd("%s/%s/SKIPPED.txt" % (sdir, arm))
        if skip:
            speed[arm]["skipped"] = skip.strip()

receipt = {
    "schema": "fn-ep-gate-receipt-v1",
    "started": E.get("GATE_STARTED"),
    "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "verdict": E.get("GATE_VERDICT") or "ERROR",
    "reason": E.get("GATE_REASON") or None,
    "config": {
        "n": n, "ref_ranks": ref_ranks, "ref_arm": ref_arm, "arch": E.get("GATE_ARCH"), "hipfire_devices_ep": E.get("GATE_DEVS"),
        "devices_source": E.get("GATE_DEVS_SOURCE"),
        "model": E.get("GATE_MODEL"), "tokens": int(E.get("GATE_TOKENS", "0") or 0),
        "max_seq": int(E.get("GATE_MAX_SEQ", "0") or 0), "state_steps": E.get("GATE_STATE_STEPS"),
        "speed_reps": int(E.get("GATE_REPS", "0") or 0), "isolated_home": E.get("GATE_NO_ISOLATE") != "1",
        "extra_env": E.get("GATE_EXTRA_ENV") or None,
        "long_prompt_truncated_to": int(E.get("GATE_LONG_TOKENS", "0") or 0),
        "oracle_arms_env": "HIPFIRE_GRAPH=0 (EP is eager; %s pinned to the same schedule)" % ("single" if ref_ranks == 1 else "the reference"),
        "speed_arms_env": "default graph setting; EP arms run eager",
    },
    "env": env,
    "link_table": link_table,
    "p2p_links_kfd": p2p,
    "build_rc": int(E["GATE_BUILD_RC"]) if E.get("GATE_BUILD_RC", "").lstrip("-").isdigit() else None,
    "arms": arms,
    "compare": compare,
    "speed": speed,
}
if mode == "reserve":
    evaluations = []
    for row in (rd(out + "/reserve-eval.jsonl") or "").splitlines():
        try:
            item = json.loads(row)
        except ValueError:
            continue
        doc = rdj(out + "/" + item["json"]) if item.get("json") else None
        if isinstance(doc, dict):
            item["verdict"] = doc.get("verdict")
            item["reasons"] = doc.get("reasons") or ([doc["first_mismatch"]["message"]] if doc.get("first_mismatch") else [])
            item["summary"] = doc.get("summary")
        evaluations.append(item)
    receipt["mode"] = "reserve"
    receipt["reserve"] = {
        "switch": "HIPFIRE_EP_RESERVE_DECOUPLE", "off_arm": "=0", "on_arm": "=1", "arms": arm_names,
        "prompt_chunk_rows": 4096, "min_resident_layers": int(E.get("GATE_MIN_RESIDENT", "0") or 0),
        "gates_evaluated": sorted({i.get("gate") for i in evaluations}), "evaluations": evaluations,
        "performance_claim": None,
    }
    receipt["config"].update({"mode": "reserve", "mtp": mtp})
if mode == "pp":
    _tmeta = (arms.get(target_arm) or {}).get("meta") or {}
    _counts = _tmeta.get("band_counts") or []
    _by_pci = {str(r["bdf"]).lower(): r for r in link_table}
    ranks_out = []
    for row in (_tmeta.get("rank_order") or []):
        link = _by_pci.get(str(row.get("pci_bus_id") or "").lower(), {})
        rk = row.get("rank")
        ranks_out.append({
            "rank": rk, "device_id": row.get("device_id"), "arch": row.get("arch"), "pci_bus_id": row.get("pci_bus_id"),
            "layers": _counts[rk] if isinstance(rk, int) and 0 <= rk < len(_counts) else None,
            "output_owner": rk == _tmeta.get("output_owner"),
            "hipfire_device_position": link.get("hipfire_device_position"),
            "link_speed": link.get("link_speed"), "link_width": link.get("link_width"),
            "max_link_speed": link.get("max_link_speed"), "max_link_width": link.get("max_link_width"),
        })
    _mem = re.search(r"MemTotal:\s+(\d+) kB", env.get("meminfo") or "")
    receipt["mode"] = "pp"
    receipt["pp"] = n
    receipt["bands"] = {arm: (arms.get(arm) or {}).get("band_counts") for arm in arm_names}
    receipt["transport"] = {
        "requested": E.get("GATE_TRANSPORT"),
        "edges": {arm: (arms.get(arm) or {}).get("edge_transports") for arm in arm_names},
    }
    receipt["ranks_output_last"] = ranks_out or link_table
    receipt["host_ram_mib"] = int(_mem.group(1)) // 1024 if _mem else None
    receipt["config"].update({
        "mode": "pp", "pp": n, "ref_pp": ref_ranks, "target_arm": target_arm,
        "pp_layers_requested": E.get("GATE_PP_LAYERS") or None,
        "transport_requested": E.get("GATE_TRANSPORT"),
        "mtp": mtp,
        "mtp_pinned_env": "HIPFIRE_MTP_AR_FLOOR=0 HIPFIRE_MTP_INCREMENTAL=0 HIPFIRE_MTP_NGRAM=off; oracle --spec mtp --temp 0 --ngram off" if mtp else None,
        "oracle_arms_env": "HIPFIRE_GRAPH=0; every arm runs fn_ep_oracle --canonical-state and exports HIPFIRE_TOPOLOGY_PP (1 for a single-card reference); PP arms add --pp and HIPFIRE_MESH_TRANSPORT (and HIPFIRE_PP_LAYERS on the PP-N arm only)",
        "speed_arms_env": "default graph setting; every arm exports HIPFIRE_TOPOLOGY_PP and topology.pp in the private home (1 for a single-card reference)",
        "p2p_links_order": "HIPFIRE_DEVICES position",
    })
    receipt["speed_pp_assertion_failed"] = E.get("GATE_SPEED_FAIL") == "1"
    if mtp:
        receipt["compare_mtp"] = compare_mtp
with open(out + "/receipt.json", "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2)
    handle.write("\n")

# ------------------------------------------------------------- summary.txt
lines = []
lines.append("fn-ep-gate: %s%s" % (receipt["verdict"], " (%s)" % receipt["reason"] if receipt["reason"] else ""))
if mode == "pp":
    lines.append("mode=pp  arch=%s  pp=%s  hipfire devices (reference=first %d, output card first): %s" % (E.get("GATE_ARCH"), n, ref_ranks, E.get("GATE_DEVS")))
    lines.append("transport=%s  requested bands=%s  reference: %s" % (
        E.get("GATE_TRANSPORT"), E.get("GATE_PP_LAYERS") or "uniform",
        "one card (host-mapped experts by default)" if ref_ranks == 1 else "PP%d on the first %d cards, uniform bands" % (ref_ranks, ref_ranks)))
    lines.append("ref_ranks=%d  mtp exactness pair: %s" % (ref_ranks, "on" if mtp else "off"))
elif mode == "reserve":
    lines.append("mode=reserve  arch=%s  EP%s  hipfire devices: %s  (off arm: HIPFIRE_EP_RESERVE_DECOUPLE=0; on arm: HIPFIRE_EP_RESERVE_DECOUPLE=1; one binary)" % (E.get("GATE_ARCH"), n, E.get("GATE_DEVS")))
    lines.append("prompt chunk rows 4096 (never lowered)  min resident layers/rank %s  arms: %s" % (E.get("GATE_MIN_RESIDENT"), " ".join(arm_names)))
    for item in receipt["reserve"]["evaluations"]:
        lines.append("  %-16s %-10s %s" % (item.get("gate"), item.get("verdict") or item.get("state"), "; ".join(item.get("reasons") or [])[:300]))
    if not any(str(i.get("gate")).startswith(("perf", "c1")) for i in receipt["reserve"]["evaluations"]):
        lines.append("  Gate 2 (pp32K) and Gate 3 (weighted C1) were NOT evaluated: no --perf-json / --c1-manifest supplied. No performance claim.")
else:
    if ref_ranks == 1:
        lines.append("arch=%s  N=%s  hipfire devices (single=first, EP=all): %s" % (E.get("GATE_ARCH"), n, E.get("GATE_DEVS")))
    else:
        lines.append("arch=%s  N=%s  hipfire devices (reference=first %d, EP=all): %s" % (E.get("GATE_ARCH"), n, ref_ranks, E.get("GATE_DEVS")))
        lines.append("reference: EP%d on the first %d cards (one card = --ref-ranks 1)" % (ref_ranks, ref_ranks))
    lines.append("ref_ranks=%d" % ref_ranks)
    lines.append("Note: EP-N ≡ one card is established separately by our gates.")
lines.append("model: %s" % E.get("GATE_MODEL"))
git = (env["git"] or "").strip().splitlines()
lines.append("build: %s" % (git[0] if git else "unknown"))
_prompts = (arms.get(ref_arm) or {}).get("prompts") or []
if _prompts:
    lines.append("prompts (tokens used / full): " + ", ".join(
        "%s/%s" % (p.get("token_count"), p.get("full_token_count")) for p in _prompts))
lines.append("")
lines.append("rank idx bdf          arch    vram_MiB  link (cur / max)")
for row in link_table:
    lines.append("%-4d %-3d %-12s %-7s %-9d %s x%s / %s x%s" % (
        row["rank"], row["index"], row["bdf"], row["arch"], row["vram_total_mib"],
        row["link_speed"], row["link_width"], row["max_link_speed"], row["max_link_width"]))
lines.append("")
if mode == "pp":
    for arm in arm_names:
        info = arms.get(arm) or {}
        if info.get("meta"):
            lines.append("placement %-11s pp=%s bands=%s output_owner=%s edges=%s" % (
                arm, info.get("pp"), info.get("band_counts"), info.get("output_owner"),
                ",".join("%s>%s:%s" % (e.get("src"), e.get("dst"), e.get("transport")) for e in (info.get("edge_transports") or [])) or "none"))
    lines.append("")
for arm, info in arms.items():
    status = "ok" if info["rc"] == 0 else "FAILED rc=%s (%s)" % (info["rc"], (info["failure"] or {}).get("kind"))
    lines.append("arm %-7s %s  %ss  vram peak MiB/card: %s" % (arm, status, info["seconds"], info["vram_peak_mib_per_card"]))
    if info["failure"]:
        lines.append("    stderr tail: " + "\n    ".join((info["failure"]["stderr_tail"] or "").splitlines()[-6:]))
lines.append("")
lines.append("compare: %s" % (compare.get("summary") or compare.get("state")))
if compare.get("first_mismatch"):
    lines.append("    first mismatch: %s" % compare["first_mismatch"].get("message"))
if compare_mtp is not None:
    lines.append("compare (mtp): %s" % (compare_mtp.get("summary") or compare_mtp.get("state")))
    if compare_mtp.get("first_mismatch"):
        lines.append("    first mismatch: %s" % compare_mtp["first_mismatch"].get("message"))
lines.append("")
if speed:
    lines.append("speed (median of fresh processes, tok/s)       decode   prefill(4,400-token prompt)")
    base_d = (speed.get(ref_arm, {}).get("decode") or {}).get("median")
    base_p = (speed.get(ref_arm, {}).get("prefill") or {}).get("median")
    for arm, cells in speed.items():
        d = (cells.get("decode") or {}).get("median")
        p = (cells.get("prefill") or {}).get("median")
        fmt = lambda v: "n/a" if v is None else "%.1f" % v
        rel = lambda v, b: "" if v is None or not b else " (%.2fx)" % (v / b)
        lines.append("  %-7s n_ok=%s/%s  %s%s   %s%s%s" % (
            arm, (cells.get("decode") or {}).get("n_ok"), (cells.get("prefill") or {}).get("n_ok"),
            fmt(d), rel(d, base_d) if arm != ref_arm else "", fmt(p), rel(p, base_p) if arm != ref_arm else "",
            "   [%s]" % cells["skipped"] if cells.get("skipped") else ""))
lines.append("")
lines.append("Send back: receipt.json, summary.txt, %s, gate.log (and the oracle/*-receipt dirs on a failure)." % ("compare*.txt" if mode in ("pp", "reserve") else "compare.txt"))
with open(out + "/summary.txt", "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
PY
}

on_exit() {
  local rc=$?
  set +e
  [[ -z $SAMPLER ]] || { kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; }
  if ((RECEIPT_DONE == 0)); then
    [[ -n $VERDICT ]] || { VERDICT=ERROR; [[ -n $REASON ]] || REASON="aborted (exit $rc)"; }
    write_receipt
  fi
  trap - EXIT
  exit "${FINAL_RC:-$rc}"
}
trap on_exit EXIT

# --------------------------------------------------------- (a) environment
collect_env() {
  local d=$OUT/env
  {
    echo "date_utc=$(date -u +%FT%TZ)"
    echo "host=$(hostname 2>/dev/null || true)"
    echo "kernel=$(uname -srvm)"
    [[ ! -r /etc/os-release ]] || grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
  } > "$d/host.txt" 2>&1 || true
  grep -E '^(MemTotal|MemAvailable|SwapTotal):' /proc/meminfo > "$d/meminfo.txt" 2>&1 || true
  { lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Socket|Thread|NUMA node\(s\))' || true; } > "$d/cpu.txt"
  cat /proc/cmdline > "$d/cmdline.txt" 2>&1 || true
  {
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "git: unknown"
    git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true
    echo "dirty_files=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l)"
  } > "$d/git.txt" 2>&1 || true
  local rocm="" p
  for p in "${ROCM_PATH:-}" /opt/rocm /opt/rocm-*; do
    [[ -n $p && -r $p/.info/version ]] && { rocm=$p; break; }
  done
  {
    echo "rocm_path=${rocm:-unknown}"
    [[ -z $rocm ]] || echo "rocm_version=$(cat "$rocm/.info/version")"
  } > "$d/rocm-version.txt" 2>&1 || true
  { hipcc --version 2>&1 || "${rocm:-/opt/rocm}/bin/hipcc" --version 2>&1 || echo "hipcc: not found"; } > "$d/hipcc-version.txt" 2>&1 || true
  {
    echo "amdgpu_module_version=$(cat /sys/module/amdgpu/version 2>/dev/null || echo unknown)"
    command -v rocm-smi >/dev/null && rocm-smi --showdriverversion 2>&1 || true
  } > "$d/driver.txt" 2>&1 || true
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showtopo --showtopoaccess > "$d/rocm-smi-topo.txt" 2>&1 || echo "rocm-smi --showtopo --showtopoaccess failed" >> "$d/rocm-smi-topo.txt"
  else
    echo "rocm-smi not found" > "$d/rocm-smi-topo.txt"
  fi
  { env | grep -E '^(HIP|HIPFIRE|ROCR|HSA|GPU_|AMD_|ROC_)' | sort || true; } > "$d/inherited-env.txt"

  python3 - "$d/cards.json" <<'PY'
import json, os, sys

KFD = "/sys/class/kfd/kfd/topology/nodes"


def rd(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def props(path):
    out = {}
    for line in (rd(path) or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1].strip()
    return out


def ival(d, key):
    try:
        return int(d.get(key, 0))
    except ValueError:
        return 0


cards = []
if os.path.isdir(KFD):
    for name in sorted((n for n in os.listdir(KFD) if n.isdigit()), key=int):
        p = props("%s/%s/properties" % (KFD, name))
        if ival(p, "simd_count") == 0:
            continue
        version, loc, dom = ival(p, "gfx_target_version"), ival(p, "location_id"), ival(p, "domain")
        bdf = "%04x:%02x:%02x.%x" % (dom, loc >> 8, (loc >> 3) & 0x1F, loc & 7)
        sysfs = "/sys/bus/pci/devices/" + bdf
        banks = []
        bank_dir = "%s/%s/mem_banks" % (KFD, name)
        if os.path.isdir(bank_dir):
            for bank in sorted(os.listdir(bank_dir)):
                bp = props("%s/%s/properties" % (bank_dir, bank))
                banks.append({"heap_type": ival(bp, "heap_type"), "size_in_bytes": ival(bp, "size_in_bytes")})
        links = []
        for kind in ("p2p_links", "io_links"):
            ldir = "%s/%s/%s" % (KFD, name, kind)
            if os.path.isdir(ldir):
                for link in sorted(os.listdir(ldir)):
                    lp = props("%s/%s/properties" % (ldir, link))
                    if lp:
                        links.append({"kind": kind, "to_node": ival(lp, "node_to"), "type": ival(lp, "type"),
                                      "weight": ival(lp, "weight"), "flags": ival(lp, "flags")})
        chain = [seg for seg in os.path.realpath(sysfs).split("/") if len(seg) == 12 and seg[4] == ":"]
        vram_total = rd(sysfs + "/mem_info_vram_total")
        cards.append({
            "node": int(name),
            "bdf": bdf,
            "arch": "gfx%d%x%x" % (version // 10000, (version // 100) % 100, version % 100),
            "gfx_target_version": version,
            "kfd_name": rd("%s/%s/name" % (KFD, name)),
            "unique_id": p.get("unique_id"),
            "vendor_id": ival(p, "vendor_id"), "device_id": ival(p, "device_id"),
            "kfd_mem_banks": banks,
            "vram_total_bytes": int(vram_total) if vram_total and vram_total.isdigit() else
                max([b["size_in_bytes"] for b in banks] or [0]),
            "current_link_speed": rd(sysfs + "/current_link_speed"),
            "current_link_width": rd(sysfs + "/current_link_width"),
            "max_link_speed": rd(sysfs + "/max_link_speed"),
            "max_link_width": rd(sysfs + "/max_link_width"),
            "pci_chain": chain,
            "p2p_links": [l for l in links if l["kind"] == "p2p_links"],
            "io_links": [l for l in links if l["kind"] == "io_links"],
        })
cards.sort(key=lambda c: c["bdf"])
for index, card in enumerate(cards):
    card["index"] = index
json.dump({"kfd_present": os.path.isdir(KFD), "cards": cards}, open(sys.argv[1], "w"), indent=2)
print("fn-ep-gate: %d GPU node(s): %s" % (len(cards), ", ".join("%d:%s@%s" % (c["index"], c["arch"], c["bdf"]) for c in cards)))
PY
}

select_cards() {
  local sel
  sel=$(python3 - "$OUT/env/cards.json" "$ARCH" "$N" "$DEVLIST" <<'PY'
import collections, json, sys

doc = json.load(open(sys.argv[1]))
cards, want, n, devlist = doc["cards"], sys.argv[2], sys.argv[3], sys.argv[4]
SUPPORTED = ("gfx1100", "gfx1201")


def fail(message):
    print("fn-ep-gate: " + message, file=sys.stderr)
    sys.exit(2)


def table():
    return "visible cards: " + "; ".join("%d:%s@%s uuid=%s" % (c["index"], c["arch"], c["bdf"],
        ("GPU-%016x" % int(c["unique_id"])) if str(c.get("unique_id") or "0").isdigit() and int(c.get("unique_id") or 0) else "none")
        for c in cards)


def parse_bdf(text):
    # DDDD:BB:DD.F or BB:DD.F, hex; returns (domain or None, bus, dev, fn)
    try:
        head, fn = text.rsplit(".", 1)
        parts = head.split(":")
        if len(parts) == 3:
            return (int(parts[0], 16), int(parts[1], 16), int(parts[2], 16), int(fn, 16))
        if len(parts) == 2:
            return (None, int(parts[0], 16), int(parts[1], 16), int(fn, 16))
    except ValueError:
        pass
    return None


def card_bdf(c):
    d, b, rest = c["bdf"].split(":")
    dev, fn = rest.split(".")
    return (int(d, 16), int(b, 16), int(dev, 16), int(fn, 16))


if not doc["kfd_present"] or not cards:
    fail("no GPU found in the KFD topology (/sys/class/kfd/kfd/topology/nodes)")

if devlist:
    # Resolve every entry of a hardware.devices-style list to a KFD card, keeping list order.
    entries = [e.strip() for e in devlist.split(",")]
    chosen, deferred = [None] * len(entries), []
    for slot, entry in enumerate(entries):
        low = entry.lower()
        if not entry:
            fail("empty entry in --devices %r" % devlist)
        if entry.isdigit():
            hits = [c for c in cards if c["index"] == int(entry)]
        elif low.startswith("gpu-"):
            try:
                uid = int(low[4:], 16)
            except ValueError:
                fail("%r is not a GPU-<hex> uuid" % entry)
            hits = [c for c in cards if str(c.get("unique_id") or "0").isdigit() and int(c.get("unique_id") or 0) == uid and uid]
        elif low.startswith("gfx"):
            deferred.append(slot)
            continue
        elif ":" in entry:
            bdf = parse_bdf(entry)
            if bdf is None:
                fail("%r is not a DDDD:BB:DD.F or BB:DD.F PCI address" % entry)
            hits = [c for c in cards if card_bdf(c)[1:] == bdf[1:] and (bdf[0] is None or card_bdf(c)[0] == bdf[0])]
        else:
            fail("unknown device selector %r (index, GPU-uuid, PCI address or gfxNNNN); %s" % (entry, table()))
        if len(hits) != 1:
            fail("--devices entry %r matches %d card(s); %s" % (entry, len(hits), table()))
        chosen[slot] = hits[0]
    for slot in deferred:
        taken = {c["index"] for c in chosen if c}
        hits = [c for c in cards if c["arch"] == entries[slot].lower() and c["index"] not in taken]
        if not hits:
            fail("--devices entry %r: no unused %s card; %s" % (entries[slot], entries[slot], table()))
        chosen[slot] = hits[0]
    if len({c["index"] for c in chosen}) != len(chosen):
        fail("--devices names the same card more than once; %s" % table())
    count = len(chosen)
    if n and int(n) != count:
        fail("N=%s but --devices lists %d card(s)" % (n, count))
    archs = sorted({c["arch"] for c in chosen})
    if len(archs) != 1:
        print("fn-ep-gate: WARNING: --devices mixes archs %s (EP needs a homogeneous set)" % archs, file=sys.stderr)
    arch = chosen[0]["arch"]
    if want and any(c["arch"] != want for c in chosen):
        fail("--arch %s but --devices selects %s" % (want, archs))
    if not 2 <= count <= 8:
        fail("N=%d is outside 2..=8" % count)
else:
    if want:
        arch = want
    else:
        counts = collections.Counter(c["arch"] for c in cards)
        supported = [a for a in counts if a in SUPPORTED]
        if supported:
            arch = max(supported, key=lambda a: (counts[a], -min(c["index"] for c in cards if c["arch"] == a)))
        else:
            arch = cards[0]["arch"]
            print("fn-ep-gate: WARNING: no gfx1100/gfx1201 card; trying %s (EP admission will likely refuse it)" % arch,
                  file=sys.stderr)
    same = [c for c in cards if c["arch"] == arch]
    if not same:
        fail("no %s card found (have: %s)" % (arch, sorted(set(c["arch"] for c in cards))))
    count = int(n) if n else min(len(same), 8)
    if not 2 <= count <= 8:
        fail("N=%d is outside 2..=8 (%d %s card(s) visible)" % (count, len(same), arch))
    if count > len(same):
        fail("N=%d but only %d %s card(s) are visible" % (count, len(same), arch))
    chosen = same[:count]
print(arch, count, ",".join(c["bdf"] for c in chosen), ",".join(str(c["index"]) for c in chosen))
PY
) || { REASON="card selection failed (see gate.log)"; VERDICT=ERROR; FINAL_RC=2; exit 2; }
  local bdfs
  read -r ARCH N bdfs CARD_IDX <<<"$sel"
  IFS=, read -r -a BDFS <<<"$bdfs"
  [[ -n $DEVLIST ]] && DEVS=$DEVLIST || DEVS=$CARD_IDX
  SINGLE_DEVS=${DEVS%%,*}
  log "arch=$ARCH N=$N devices(HIPFIRE_DEVICES)=$DEVS (source: $DEVS_SOURCE) single=$SINGLE_DEVS cards(PCI-order index)=$CARD_IDX bdfs=$bdfs"
  if ((REF_RANKS < 1 || REF_RANKS >= N)); then
    REASON="--ref-ranks $REF_RANKS is outside 1..$((N - 1)) for N=$N"; VERDICT=ERROR; FINAL_RC=2
    die "$REASON"
  fi
  if ((REF_RANKS == 1)); then
    REF_ARM=single REF_DEVS=$SINGLE_DEVS
  else
    local _devs
    IFS=, read -r -a _devs <<<"$DEVS"
    REF_ARM=ref-$MODE$REF_RANKS
    REF_DEVS=$(IFS=,; echo "${_devs[*]:0:REF_RANKS}")
    log "reference arm $REF_ARM: $MODE=$REF_RANKS devices=$REF_DEVS"
  fi
  if [[ $MODE == pp && -n $PP_LAYERS ]]; then
    local _counts
    IFS=, read -r -a _counts <<<"$PP_LAYERS"
    if ((${#_counts[@]} != N)); then
      REASON="--pp-layers lists ${#_counts[@]} entries, expected N=$N"; VERDICT=ERROR; FINAL_RC=2
      die "$REASON"
    fi
  fi
}

hash_model() {
  {
    echo "path=$MODEL"
    echo "size_bytes=$(stat -c %s "$MODEL")"
    if ((SKIP_SHA)); then
      echo "sha256=skipped"
    else
      log "hashing model (use --skip-model-sha to skip)"
      echo "sha256=$(sha256sum "$MODEL" | cut -d' ' -f1)"
    fi
  } > "$OUT/env/model.txt"
}

# ------------------------------------------------------------------ (b) build
build_all() {
  if ((SKIP_BUILD)); then
    log "build skipped"
    BUILD_RC=0
  else
    log "building fn_ep_oracle, hipfire, daemon (log: $OUT/build.log)"
    BUILD_RC=0
    (
      cd "$REPO_ROOT"
      cargo build --release -p hipfire-loader --example fn_ep_oracle --features lab \
        && cargo build --release -p hipfire-cli -p hipfire-daemon
    ) > "$OUT/build.log" 2>&1 || BUILD_RC=$?
  fi
  if [[ $BUILD_RC != 0 ]]; then
    REASON="cargo build failed (rc=$BUILD_RC); see build.log"; VERDICT=ERROR; FINAL_RC=2
    tail -n 20 "$OUT/build.log" || true
    exit 2
  fi
  local missing=""
  for bin in "$ORACLE" "$HIPFIRE" "$DAEMON"; do [[ -x $bin ]] || missing+=" $bin"; done
  [[ -z $missing ]] || { REASON="missing binaries:$missing"; VERDICT=ERROR; FINAL_RC=2; die "$REASON"; }
  { "$HIPFIRE" version 2>&1 | sed -n '1,25p'; } > "$OUT/env/hipfire-version.txt" 2>&1 || true
}

# --------------------------------------------------------------- arm helpers
ARM_ENV=()
arm_env() {
  local arm=$1 ep=$2 devs=$3 pp=${4:-0} mtp=${5:-0} layers=${6:-} reserve=${7:-0}
  # ROCR_VISIBLE_DEVICES: dropped only when the card list came from our own KFD selection; a list the
  # caller supplied (--devices) or inherited (a scheduler's HIPFIRE_DEVICES) leaves ROCR_VISIBLE_DEVICES alone.
  local unset_rocr=() unset_pp=()
  [[ $DEVS_SOURCE != kfd ]] || unset_rocr=(-u ROCR_VISIBLE_DEVICES)
  # PP mode: everything an arm pins, and every placement variable the loader refuses under pp, is cleared first so
  # an inherited value cannot leak into (or break) an arm.
  if [[ $MODE == pp ]]; then
    unset_pp=(-u HIPFIRE_PP_LAYERS -u HIPFIRE_MESH_TRANSPORT -u HIPFIRE_TOPOLOGY_PP
      -u HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS -u HIPFIRE_QWEN4_EXPERT_HOTSET -u HIPFIRE_QWEN4_EXPERT_STAGE
      -u HIPFIRE_QWEN4_TURN_SNAPSHOTS -u HIPFIRE_MTP_AR_FLOOR -u HIPFIRE_MTP_INCREMENTAL -u HIPFIRE_MTP_NGRAM)
  fi
  if [[ $MODE == reserve ]]; then
    # Reserve arms: the switch is set last (=0 off, =1 on), the chunk override is cleared so the prompt chunk stays 4096.
    unset_pp=(-u HIPFIRE_EP_RESERVE_DECOUPLE -u HIPFIRE_EP_RESERVE_RECEIPT -u HIPFIRE_PREFILL_CHUNK_ROWS
      -u HIPFIRE_MTP_AR_FLOOR -u HIPFIRE_MTP_INCREMENTAL -u HIPFIRE_MTP_NGRAM)
  fi
  ARM_ENV=(env -u HIP_VISIBLE_DEVICES -u HIPFIRE_QWEN4_EP ${unset_rocr[@]+"${unset_rocr[@]}"} ${unset_pp[@]+"${unset_pp[@]}"}
    "HIPFIRE_DEVICES=$devs" "HIPFIRE_TOPOLOGY_EP=$ep" HIPFIRE_NO_REGISTRY_FETCH=1 "HIPFIRE_DAEMON_BIN=$DAEMON")
  if [[ $MODE == pp ]]; then
    # Every PP-mode arm, the single-card reference included, pins its degree so an inherited topology.pp cannot leak in.
    ((pp >= 1)) || pp=1
    ARM_ENV+=("HIPFIRE_TOPOLOGY_PP=$pp")
    if ((pp > 1)); then
      ARM_ENV+=("HIPFIRE_MESH_TRANSPORT=$TRANSPORT")
      [[ -z $layers ]] || ARM_ENV+=("HIPFIRE_PP_LAYERS=$layers")
    fi
  fi
  if ((NO_ISOLATE == 0)); then
    local home=$OUT/home-$arm
    mkdir -p "$home/.hipfire"
    if [[ $MODE == pp ]]; then
      printf '[topology]\nep = %s\npp = %s\n' "$ep" "$pp" > "$home/.hipfire/config.toml"
    else
      printf '[topology]\nep = %s\n' "$ep" > "$home/.hipfire/config.toml"
    fi
    ARM_ENV+=("HOME=$home" "HIPFIRE_HOME=$home/.hipfire")
  fi
  [[ -n ${HIPFIRE_KERNEL_CACHE:-} ]] || ARM_ENV+=("HIPFIRE_KERNEL_CACHE=$OUT/kcache")
  if [[ -n $EXTRA_ENV ]]; then
    # shellcheck disable=SC2206
    local extra=($EXTRA_ENV)
    ARM_ENV+=("${extra[@]}")
  fi
  # Every reserve arm prints receipts (the off arm's come from its ordinary load); only the on arms switch decoupling on.
  [[ $MODE != reserve ]] || ARM_ENV+=(HIPFIRE_EP_RESERVE_RECEIPT=1)
  # The route is default on: a reserve-mode off arm sets the kill switch explicitly.
  if ((reserve != 0)); then
    ARM_ENV+=(HIPFIRE_EP_RESERVE_DECOUPLE=1)
  elif [[ $MODE == reserve ]]; then
    ARM_ENV+=(HIPFIRE_EP_RESERVE_DECOUPLE=0)
  fi
  # The native-MTP policy is pinned last so EXTRA_ENV cannot change the window schedule of an exactness arm.
  ((mtp == 0)) || ARM_ENV+=("${PP_PINNED_MTP[@]}")
}

start_vram_sampler() {
  local file=$1 count=$2
  (
    while :; do
      line=$(date +%s)
      for ((i = 0; i < count; i++)); do
        line+=" $(cat "/sys/bus/pci/devices/${BDFS[$i]}/mem_info_vram_used" 2>/dev/null || echo 0)"
      done
      echo "$line"
      sleep 1
    done
  ) > "$file" 2>/dev/null &
  SAMPLER=$!
}

stop_vram_sampler() {
  [[ -z $SAMPLER ]] || { kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true; SAMPLER=""; }
}

run_oracle_arm() {
  local arm=$1 ep=$2 devs=$3 count=$4 pp=${5:-0} mtp=${6:-0} layers=${7:-} reserve=${8:-0}
  local rec=$OUT/oracle/$arm-receipt out=$OUT/oracle/$arm rc=0
  rm -rf "$rec" "$out"
  mkdir -p "$rec"
  arm_env "$arm" "$ep" "$devs" "$pp" "$mtp" "$layers" "$reserve"
  local oracle_extra=()
  if [[ $MODE == pp ]]; then
    oracle_extra+=(--canonical-state)
    ((pp <= 1)) || oracle_extra+=(--pp "$pp")
  fi
  if [[ $MODE == pp || $MODE == reserve ]]; then
    ((mtp == 0)) || oracle_extra+=(--spec mtp --temp 0 --ngram off)
  fi
  [[ $MODE != reserve ]] || oracle_extra+=(--seed "$SEED")
  local cmd=("${ARM_ENV[@]}" HIPFIRE_GRAPH=0 timeout -k 30 "$ARM_TIMEOUT" "$ORACLE"
    --model "$MODEL" --max-seq "$MAX_SEQ" --out "$out" --arm "$arm"
    "${PROMPT_ARGS[@]}"
    --tokens "$TOKENS" --state-steps "$STATE_STEPS" --skip-model-hash ${oracle_extra[@]+"${oracle_extra[@]}"})
  printf '%q ' "${cmd[@]}" > "$rec/command.txt"
  echo >> "$rec/command.txt"
  date -u +%FT%TZ > "$rec/start.txt"
  start_vram_sampler "$rec/vram.txt" "$count"
  log "oracle arm $arm: ep=$ep devices=$devs"
  [[ $MODE != pp ]] || log "oracle arm $arm: pp=$pp mtp=$mtp layers=${layers:-uniform}"
  "${cmd[@]}" > "$rec/stdout.txt" 2> "$rec/stderr.txt" || rc=$?
  stop_vram_sampler
  date -u +%FT%TZ > "$rec/end.txt"
  echo "$rc" > "$rec/rc.txt"
  log "oracle arm $arm: rc=$rc"
  if ((rc != 0)); then
    log "last stderr lines of arm $arm:"
    tail -n 8 "$rec/stderr.txt" | sed 's/^/    /' || true
  fi
  return 0
}

run_bench_cell() {
  local arm=$1 ep=$2 devs=$3 cell=$4 rep=$5 prompt=$6 max_tokens=$7 pp=${8:-0} layers=${9:-}
  local dir=$OUT/speed/$arm base rc=0
  mkdir -p "$dir"
  base=$dir/$cell-$rep
  arm_env "$arm" "$ep" "$devs" "$pp" 0 "$layers"
  local cmd=("${ARM_ENV[@]}" timeout -k 30 "$BENCH_TIMEOUT" "$HIPFIRE" bench "$MODEL"
    --spec off --max-seq "$MAX_SEQ" --runs 1 --max-tokens "$max_tokens" --prompt-file "$prompt" -j)
  printf '%q ' "${cmd[@]}" > "$base.cmd"
  echo >> "$base.cmd"
  "${cmd[@]}" > "$base.json" 2> "$base.stderr" || rc=$?
  if [[ $MODE == pp ]] && ((rc == 0)); then
    # The cell only counts when the bench's daemon really loaded the requested pipeline degree.
    local want=$pp
    ((want >= 1)) || want=1
    if ! python3 "$OUT/pp_bench_check.py" "$base.json" "$want" > "$base.pp" 2>&1; then
      rc=97
      cat "$base.pp" >> "$base.stderr"
      SPEED_FAIL=1
    fi
  fi
  # A failed cell (bench error or pp mismatch) must keep the PP gate from passing.
  [[ $MODE != pp ]] || ((rc == 0)) || SPEED_FAIL=1
  echo "$rc" > "$base.rc"
  log "bench $arm $cell rep $rep: rc=$rc"
  return 0
}

oracle_arm_ok() {
  local arm=$1
  [[ $(cat "$OUT/oracle/$arm-receipt/rc.txt" 2>/dev/null || echo x) == 0 ]] && [[ -f $OUT/oracle/$arm/meta.json ]]
}

# ---------------------------------------------------------------- PP mode helpers
write_pp_helpers() {
  cat > "$OUT/pp_compare.py" <<'PPCMP'
#!/usr/bin/env python3
"""PP comparator written by fn-ep-gate.sh --mode pp (stdlib only).

    pp_compare.py REF_DIR PP_DIR --base fn_ep_compare.py --ref-pp R --pp N --ref-bdfs LIST --bdfs LIST --arch ARCH
                  [--layers COUNTS] [--transport host|auto] [--mtp 0|1] [--json OUT.json]

Both directories are fn_ep_oracle --canonical-state outputs (reference: pp = 1 or PP-R; target: PP-N). The
existing comparator (fn_ep_compare.py) requires EP rank mirrors and cannot read window/tokens records, so this
PP-only comparator imports its Verdict / read_jsonl / read_meta / raw-logits helpers and compares, as
ep-host-notes/g0/g0_compare.py does:
  * meta: the `ranks` field is ignored only when both metas carry canonical_state == true and ranks == pp
    (default 1); completed, tokens, max_seq, spec, steps_run, prompts, state_steps and vocab must be equal;
  * every ordered record of logits.jsonl and state.jsonl, field for field (logits, window, tokens, state);
  * the .f32 dump set and every dump byte for byte;
and asserts each arm's meta.json records the requested pp and placement. Exit 0 PASS, 1 FAIL, 2 usage."""
import argparse
import filecmp
import importlib.util
import json
import os
import re
import sys

MTP_PINNED = {"HIPFIRE_MTP_AR_FLOOR": "0", "HIPFIRE_MTP_INCREMENTAL": "0", "HIPFIRE_MTP_NGRAM": "off"}


def load_base(path):
    spec = importlib.util.spec_from_file_location("fn_ep_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pci(value):
    return str(value or "").lower()


def expected_env(pp, layers, transport, mtp):
    env = {
        "HIPFIRE_PP_LAYERS": layers or None,
        "HIPFIRE_TOPOLOGY_PP": str(pp),
        "HIPFIRE_MESH_TRANSPORT": transport if pp > 1 else None,
    }
    for key, value in MTP_PINNED.items():
        env[key] = value if mtp else None
    return env


def check_arm(v, label, meta, pp, bdfs, arch, layers, transport, mtp):
    def check(ok, text):
        v.checked("placement")
        if not ok:
            v.fail("placement", "%s: %s" % (label, text))

    check(meta.get("canonical_state") is True, "meta.canonical_state is %r, expected true" % (meta.get("canonical_state"),))
    check(meta.get("pp", 1) == pp, "meta.pp is %r, requested %d" % (meta.get("pp", 1), pp))
    check(meta.get("ranks") == pp, "meta.ranks is %r, requested %d" % (meta.get("ranks"), pp))
    devices = meta.get("devices") or []
    check(len(devices) == pp, "meta.devices lists %d device(s), requested %d" % (len(devices), pp))
    check(all(d.get("arch") == arch for d in devices), "meta.devices archs %r, expected %s" % ([d.get("arch") for d in devices], arch))
    env = meta.get("env") or {}
    for key, want in expected_env(pp, layers, transport, mtp).items():
        check(env.get(key) == want, "meta.env %s is %r, expected %r" % (key, env.get(key), want))
    check(bool(mtp) == (meta.get("spec") is not None), "meta.spec is %r, MTP arm expected %s" % (meta.get("spec"), bool(mtp)))
    if pp == 1:
        check(meta.get("placement") is None, "a pp = 1 arm recorded a placement: %r" % (meta.get("placement"),))
        check(bool(devices) and pci(devices[0].get("pci_bus_id")) == pci(bdfs[0]),
              "device 0 is %r, expected card %s" % ((devices[0].get("pci_bus_id") if devices else None), bdfs[0]))
        return
    placement = meta.get("placement")
    check(isinstance(placement, dict), "meta.placement missing")
    if not isinstance(placement, dict):
        return
    for key in ("pp", "band_counts", "rank_order", "edge_transports", "output_owner"):
        check(meta.get(key) == placement.get(key), "meta.%s %r differs from meta.placement.%s %r" % (key, meta.get(key), key, placement.get(key)))
    check(placement.get("pp") == pp, "placement.pp is %r, requested %d" % (placement.get("pp"), pp))
    counts = placement.get("band_counts")
    check(isinstance(counts, list) and len(counts) == pp and all(isinstance(c, int) and c > 0 for c in counts),
          "band_counts %r is not %d positive counts" % (counts, pp))
    if layers:
        check(counts == [int(x) for x in layers.split(",")], "band_counts %r != requested --pp-layers %s" % (counts, layers))
    if isinstance(counts, list) and counts and all(isinstance(c, int) for c in counts):
        if not layers:
            check(max(counts) - min(counts) <= 1, "default band_counts %r is not the uniform split" % (counts,))
        check(counts[0] >= 2, "band_counts %r: band 0 holds fewer than 2 layers (the PLE layer lives in band 0)" % (counts,))
    order = placement.get("rank_order") or []
    check(len(order) == pp, "rank_order lists %d rank(s), requested %d" % (len(order), pp))
    check([d.get("rank") for d in order] == list(range(pp)), "rank_order ranks %r" % ([d.get("rank") for d in order],))
    expected_order = [pci(b) for b in bdfs[1:] + bdfs[:1]]
    check([pci(d.get("pci_bus_id")) for d in order] == expected_order,
          "rank_order PCI ids %r != output-last rotation of the card list %r" % ([pci(d.get("pci_bus_id")) for d in order], expected_order))
    check(placement.get("output_owner") == pp - 1, "output_owner is %r, expected the last rank %d" % (placement.get("output_owner"), pp - 1))
    check(bool(order) and pci(order[-1].get("pci_bus_id")) == pci(bdfs[0]),
          "the last rank is %r, expected card 0 %s" % ((order[-1].get("pci_bus_id") if order else None), bdfs[0]))
    check(order == [dict(d, rank=i) for i, d in enumerate(devices)], "rank_order differs from devices")
    edges = placement.get("edge_transports") or []
    check(len(edges) == pp - 1, "edge_transports lists %d edge(s), expected %d" % (len(edges), pp - 1))
    for i, edge in enumerate(edges):
        check(edge.get("src") == i and edge.get("dst") == i + 1, "edge %d is %r>%r, expected %d>%d" % (i, edge.get("src"), edge.get("dst"), i, i + 1))
        check(edge.get("transport") == "host-mapped", "edge %d transport is %r, expected host-mapped" % (i, edge.get("transport")))
        check(isinstance(edge.get("epoch"), int) and edge.get("epoch") > 0, "edge %d has no live transport lease epoch" % i)
    check(isinstance(meta.get("numerical_policy"), str) and bool(meta.get("numerical_policy")), "meta.numerical_policy missing")


def compare_records(base, v, family, ref_dir, tgt_dir, name):
    try:
        a = base.read_jsonl(os.path.join(ref_dir, name))
        b = base.read_jsonl(os.path.join(tgt_dir, name))
    except (OSError, ValueError) as error:
        v.checked("record_set")
        v.fail("record_set", "cannot read %s: %s" % (name, error))
        return 0
    v.checked("record_set")
    if len(a) != len(b):
        v.fail("record_set", "%s: %d vs %d records" % (name, len(a), len(b)))
    if not a:
        v.fail("record_set", "reference %s has no records" % name)
    for i, (x, y) in enumerate(zip(a, b)):
        v.checked(family)
        if x != y:
            diff = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            text = "%s record %d (kind %s prompt %s step %s window %s): fields %s differ" % (
                name, i, x.get("kind"), x.get("prompt"), x.get("step"), x.get("window"), diff)
            detail = {"record": i, "fields": diff}
            if x.get("kind") == "logits" and isinstance(x.get("prompt"), int) and isinstance(x.get("step"), int):
                detail["raw"] = base.raw_logits_report(ref_dir, tgt_dir, x["prompt"], x["step"])
            v.fail(family, text, **detail)
    return len(a)


def compare(base, args):
    v = base.Verdict(args.max_report)
    try:
        ref_meta, tgt_meta = base.read_meta(args.ref_dir), base.read_meta(args.tgt_dir)
    except ValueError as error:
        v.fail("identity", "meta.json unreadable: %s" % error)
        return v, {}
    for label, meta in (("reference", ref_meta), ("PP", tgt_meta)):
        v.checked("identity")
        if meta is None:
            v.fail("identity", "%s arm: meta.json missing" % label)
        elif meta.get("completed") is not True:
            v.fail("identity", "%s arm: meta.json says the run did not complete" % label)
    if ref_meta is None or tgt_meta is None:
        return v, {}
    canonical_mesh = all(m.get("canonical_state") is True and m.get("ranks") == m.get("pp", 1) for m in (ref_meta, tgt_meta))
    keys = ("completed", "tokens", "max_seq", "spec", "steps_run", "prompts", "state_steps", "vocab")
    if not canonical_mesh:
        keys += ("ranks",)
    for key in keys:
        v.checked("identity")
        if ref_meta.get(key) != tgt_meta.get(key):
            v.fail("identity", "meta.%s differs: reference %r vs PP %r" % (key, ref_meta.get(key), tgt_meta.get(key)))
    sha_ref, sha_tgt = ref_meta.get("model_sha256"), tgt_meta.get("model_sha256")
    v.checked("identity")
    if sha_ref and sha_tgt and sha_ref != sha_tgt:
        v.fail("identity", "model sha256 differs: %s vs %s" % (sha_ref, sha_tgt))
    ref_bdfs = [b for b in args.ref_bdfs.split(",") if b]
    tgt_bdfs = [b for b in args.bdfs.split(",") if b]
    check_arm(v, "reference", ref_meta, args.ref_pp, ref_bdfs, args.arch, None, args.transport, args.mtp)
    check_arm(v, "PP", tgt_meta, args.pp, tgt_bdfs, args.arch, args.layers or None, args.transport, args.mtp)
    if args.ref_pp > 1:
        v.checked("placement")
        ref_counts, tgt_counts = ref_meta.get("band_counts"), tgt_meta.get("band_counts")
        if isinstance(ref_counts, list) and isinstance(tgt_counts, list) and sum(ref_counts) != sum(tgt_counts):
            v.fail("placement", "band counts cover %d layers on the reference but %d on the PP arm" % (sum(ref_counts), sum(tgt_counts)))
        v.checked("placement")
        if ref_meta.get("numerical_policy") != tgt_meta.get("numerical_policy"):
            v.fail("placement", "numerical_policy differs: %r vs %r" % (ref_meta.get("numerical_policy"), tgt_meta.get("numerical_policy")))
    logits_records = compare_records(base, v, "logits", args.ref_dir, args.tgt_dir, "logits.jsonl")
    state_records = compare_records(base, v, "state", args.ref_dir, args.tgt_dir, "state.jsonl")
    dumps_a = sorted(f for f in os.listdir(args.ref_dir) if f.endswith(".f32"))
    dumps_b = sorted(f for f in os.listdir(args.tgt_dir) if f.endswith(".f32"))
    v.checked("raw_dumps")
    if dumps_a != dumps_b:
        v.fail("raw_dumps", "raw logits dump sets differ (only in reference %s, only in PP %s)" % (
            sorted(set(dumps_a) - set(dumps_b))[:3], sorted(set(dumps_b) - set(dumps_a))[:3]))
    if not dumps_a and not args.mtp:
        v.checked("raw_dumps")
        v.fail("raw_dumps", "the reference saved no raw logits dumps (--canonical-state saves every output logits row)")
    for name in sorted(set(dumps_a) & set(dumps_b)):
        v.checked("raw_dumps")
        if not filecmp.cmp(os.path.join(args.ref_dir, name), os.path.join(args.tgt_dir, name), shallow=False):
            found = re.match(r"logits-p(\d+)-s(\d+)\.f32$", name)
            detail = {"dump": name}
            if found:
                detail["raw"] = base.raw_logits_report(args.ref_dir, args.tgt_dir, int(found.group(1)), int(found.group(2)))
            v.fail("raw_dumps", "raw dump %s differs" % name, **detail)
    shape = {"mode": "pp", "pp": args.pp, "ref_pp": args.ref_pp, "mtp": bool(args.mtp), "canonical_mesh": canonical_mesh,
             "logits_records": logits_records, "state_records": state_records, "raw_dumps": len(dumps_a)}
    return v, shape


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ref_dir")
    parser.add_argument("tgt_dir")
    parser.add_argument("--base", required=True, help="path of fn_ep_compare.py")
    parser.add_argument("--ref-pp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--ref-bdfs", required=True)
    parser.add_argument("--bdfs", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--transport", default="host")
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--json", default=None)
    parser.add_argument("--max-report", type=int, default=50)
    args = parser.parse_args(argv)
    if args.pp < 2 or args.ref_pp < 1 or args.ref_pp >= args.pp:
        print("need 1 <= --ref-pp < --pp and --pp >= 2", file=sys.stderr)
        return 2
    for d in (args.ref_dir, args.tgt_dir):
        if not os.path.isdir(d):
            print("not a directory: %s" % d, file=sys.stderr)
            return 2
    base = load_base(args.base)
    v, shape = compare(base, args)
    for name, fam in sorted(v.families.items()):
        print("%s %-10s checked=%d failed=%d" % ("PASS" if not fam["failed"] else "FAIL", name, fam["checked"], fam["failed"]))
    if v.first is not None:
        print("first mismatch [%s]: %s" % (v.first["check"], v.first["message"]))
        if v.first.get("raw"):
            print("     " + v.first["raw"])
    if v.ok:
        ref_name = "single" if args.ref_pp == 1 else "PP%d reference" % args.ref_pp
        line = ("PASS pp_compare%s: PP%d == %s, %s logits/window records, %s state records, %s raw logits dumps "
                "byte-identical (full-record equality, canonical state)" % (
                    " (mtp)" if args.mtp else "", args.pp, ref_name, shape.get("logits_records"), shape.get("state_records"), shape.get("raw_dumps")))
    else:
        failed = ", ".join("%s=%d" % (k, f["failed"]) for k, f in sorted(v.families.items()) if f["failed"])
        line = "FAIL pp_compare%s: %d failing check(s) [%s]; first: %s" % (
            " (mtp)" if args.mtp else "", sum(f["failed"] for f in v.families.values()), failed, v.first["message"])
    print(line)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({
                "verdict": "PASS" if v.ok else "FAIL",
                "summary": line,
                "reference": os.path.abspath(args.ref_dir),
                "target": os.path.abspath(args.tgt_dir),
                "shape": shape,
                "families": v.families,
                "first_mismatch": v.first,
                "mismatches": v.mismatches,
            }, handle, indent=2)
            handle.write("\n")
    return 0 if v.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
PPCMP
  cat > "$OUT/pp_bench_check.py" <<'PPCHK'
#!/usr/bin/env python3
"""pp_bench_check.py BENCH.json WANT_PP: did the bench's daemon load the requested pipeline degree?

`hipfire bench --json` reports the daemon's effective pipeline degree twice: top-level `pp` and `loaded.pp` (both
integers, from the daemon's loaded ack). Both must be present and equal WANT_PP (1 for a single-card arm).
Exit 0 ok, 1 not (missing field, mismatch or unreadable JSON)."""
import json
import sys


def doc_of(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return {}
    for start in (0, text.find("{")):
        if start < 0:
            continue
        try:
            return json.loads(text[start:])
        except ValueError:
            continue
    return {}


def as_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def main():
    doc = doc_of(sys.argv[1])
    want = int(sys.argv[2])
    top = as_int(doc.get("pp")) if isinstance(doc, dict) else None
    loaded_doc = doc.get("loaded") if isinstance(doc, dict) else None
    loaded = as_int(loaded_doc.get("pp")) if isinstance(loaded_doc, dict) else None
    print("want_pp=%d pp=%s loaded.pp=%s" % (want, top, loaded))
    if top != want or loaded != want:
        print("pp_bench_check: the bench did not report pp=%d in both `pp` and `loaded.pp` (single-GPU, a different degree, or no pp field)" % want)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
PPCHK
}

# One exactness pair: run pp_compare.py, record the state and the verdict contribution.
pp_eval_pair() {
  local label=$1 ref=$2 tgt=$3 rpp=$4 tpp=$5 mtp=$6 suffix=$7 what
  if ((rpp == 1)); then what="the single card"; else what="the PP$rpp reference"; fi
  PAIR_RC=0
  if oracle_arm_ok "$ref" && oracle_arm_ok "$tgt"; then
    local ref_bdfs tgt_bdfs
    ref_bdfs=$(IFS=,; echo "${BDFS[*]:0:rpp}")
    tgt_bdfs=$(IFS=,; echo "${BDFS[*]:0:tpp}")
    python3 "$OUT/pp_compare.py" "$OUT/oracle/$ref" "$OUT/oracle/$tgt" --base "$COMPARE" \
      --ref-pp "$rpp" --pp "$tpp" --ref-bdfs "$ref_bdfs" --bdfs "$tgt_bdfs" --arch "$ARCH" \
      --layers "$PP_LAYERS" --transport "$TRANSPORT" --mtp "$mtp" \
      --json "$OUT/compare$suffix.json" > "$OUT/compare$suffix.txt" 2>&1 || PAIR_RC=$?
    sed 's/^/    /' "$OUT/compare$suffix.txt"
    if ((PAIR_RC == 0)); then
      PAIR_STATE=PASS
    elif ((PAIR_RC == 1)); then
      PAIR_STATE=FAIL; PP_ANY_FAIL=1
      REASON+="${REASON:+; }$label: PP$tpp output differs from $what"
    else
      PAIR_STATE="comparator error rc=$PAIR_RC"; PP_ANY_ERR=1
      REASON+="${REASON:+; }$label: pp_compare.py failed to run"
    fi
  else
    PP_ANY_ERR=1
    PAIR_STATE="no comparison"
    oracle_arm_ok "$ref" || PAIR_STATE="$ref arm failed"
    oracle_arm_ok "$tgt" || PAIR_STATE="$tgt arm failed (see oracle/$tgt-receipt/stderr.txt)"
    REASON+="${REASON:+; }$label: $PAIR_STATE; no comparison"
  fi
}

run_pp_gate() {
  write_pp_helpers
  local ref_mtp=$REF_ARM-mtp tgt_mtp=$EPARM-mtp
  run_oracle_arm "$REF_ARM" 1 "$REF_DEVS" "$REF_RANKS" "$REF_RANKS" 0 ""
  run_oracle_arm "$EPARM" 1 "$DEVS" "$N" "$N" 0 "$PP_LAYERS"
  if ((MTP)); then
    run_oracle_arm "$ref_mtp" 1 "$REF_DEVS" "$REF_RANKS" "$REF_RANKS" 1 ""
    run_oracle_arm "$tgt_mtp" 1 "$DEVS" "$N" "$N" 1 "$PP_LAYERS"
  fi

  # (d) compare
  pp_eval_pair ar "$REF_ARM" "$EPARM" "$REF_RANKS" "$N" 0 ""
  COMPARE_STATE=$PAIR_STATE
  if ((MTP)); then
    pp_eval_pair mtp "$ref_mtp" "$tgt_mtp" "$REF_RANKS" "$N" 1 "-mtp"
    MTP_STATE=$PAIR_STATE
  fi
  if ((PP_ANY_ERR)); then
    VERDICT=ERROR
  elif ((PP_ANY_FAIL)); then
    VERDICT=FAIL
  else
    VERDICT=PASS
  fi

  # (e) speed cells, A/B interleaved per repetition (autoregressive, spec off)
  if ((SKIP_SPEED)); then
    log "speed cells skipped"
  else
    local tgt_speed=1
    if ! oracle_arm_ok "$EPARM"; then
      tgt_speed=0
      mkdir -p "$OUT/speed/$EPARM"
      echo "skipped: the $EPARM oracle arm failed to load/run" > "$OUT/speed/$EPARM/SKIPPED.txt"
    fi
    local rep
    for ((rep = 1; rep <= REPS; rep++)); do
      run_bench_cell "$REF_ARM" 1 "$REF_DEVS" decode "$rep" "$SHORT_PROMPT" 128 "$REF_RANKS"
      ((tgt_speed == 0)) || run_bench_cell "$EPARM" 1 "$DEVS" decode "$rep" "$SHORT_PROMPT" 128 "$N" "$PP_LAYERS"
      run_bench_cell "$REF_ARM" 1 "$REF_DEVS" prefill "$rep" "$LONG_PROMPT" 1 "$REF_RANKS"
      ((tgt_speed == 0)) || run_bench_cell "$EPARM" 1 "$DEVS" prefill "$rep" "$LONG_PROMPT" 1 "$N" "$PP_LAYERS"
    done
    if ((SPEED_FAIL)) && [[ $VERDICT == PASS ]]; then
      VERDICT=FAIL
      REASON="a speed cell failed or did not run with the requested pp (see speed/*/*.rc, *.pp, *.stderr)"
    fi
  fi
}

# ---------------------------------------------------------------- reserve mode
# One exactness pair: assemble both arms' receipts from their logs, then fn_ep_compare.py reserve arms.
# $1 gate label, $2 off arm, $3 on arm (or the repeat arm), $4 expected mode (ar|mtp), $5 extra comparator flags.
reserve_eval_pair() {
  local gate=$1 off=$2 on=$3 mode=$4 extra=${5:-} arm
  local json=compare-$gate.json
  PAIR_RC=0
  if oracle_arm_ok "$off" && oracle_arm_ok "$on"; then
    local msha
    msha=$(sed -n 's/^sha256=//p' "$OUT/env/model.txt" 2>/dev/null | head -n 1)
    [[ $msha != skipped ]] || msha=""
    for arm in "$off" "$on"; do
      cat "$OUT/oracle/$arm-receipt/stdout.txt" "$OUT/oracle/$arm-receipt/stderr.txt" > "$OUT/oracle/$arm-receipt/receipt-lines.txt" 2>/dev/null || true
      python3 "$COMPARE" reserve harvest "$OUT/oracle/$arm-receipt/receipt-lines.txt" "$OUT/oracle/$arm" ${msha:+--model-sha256 "$msha"} > "$OUT/oracle/$arm-receipt/harvest.txt" 2>&1 || true
    done
    # shellcheck disable=SC2086
    python3 "$COMPARE" reserve arms "$OUT/oracle/$off" "$OUT/oracle/$on" --expect-mode "$mode" --min-resident "$MIN_RESIDENT" \
      --min-tokens "$RESERVE_MIN_TOKENS" --require-lengths "$BOUNDARY_LENGTHS" $extra \
      --json "$OUT/$json" > "$OUT/compare-$gate.txt" 2>&1 || PAIR_RC=$?
    sed 's/^/    /' "$OUT/compare-$gate.txt"
    case $PAIR_RC in
      0) PAIR_STATE=PASS ;;
      1) PAIR_STATE=FAIL; RES_ANY_FAIL=1; REASON+="${REASON:+; }$gate: $on differs from $off or its receipts are rejected (compare-$gate.txt)" ;;
      *) PAIR_STATE="comparator error rc=$PAIR_RC"; RES_ANY_ERR=1; REASON+="${REASON:+; }$gate: fn_ep_compare.py reserve arms failed to run" ;;
    esac
  else
    RES_ANY_ERR=1
    PAIR_STATE="no comparison"
    oracle_arm_ok "$off" || PAIR_STATE="$off arm failed"
    oracle_arm_ok "$on" || PAIR_STATE="$on arm failed (see oracle/$on-receipt/stderr.txt)"
    REASON+="${REASON:+; }$gate: $PAIR_STATE; no comparison"
    json=""
  fi
  printf '{"gate":"%s","state":"%s","json":"%s"}\n' "$gate" "$PAIR_STATE" "$json" >> "$OUT/reserve-eval.jsonl"
}

# One supplied document (perf or c1): evaluate it with the kill rules.
reserve_eval_doc() {
  local gate=$1 kind=$2 file=$3 rc=0 state
  python3 "$COMPARE" reserve "$kind" "$file" --min-resident "$MIN_RESIDENT" --json "$OUT/$gate.json" > "$OUT/$gate.txt" 2>&1 || rc=$?
  sed 's/^/    /' "$OUT/$gate.txt"
  case $rc in
    0) state=PASS ;;
    1) state=FAIL; RES_ANY_FAIL=1; REASON+="${REASON:+; }$gate: kill rule or validation failed ($gate.txt)" ;;
    *) state="comparator error rc=$rc"; RES_ANY_ERR=1; REASON+="${REASON:+; }$gate: fn_ep_compare.py reserve $kind failed to run" ;;
  esac
  printf '{"gate":"%s","state":"%s","json":"%s"}\n' "$gate" "$state" "$gate.json" >> "$OUT/reserve-eval.jsonl"
}

run_reserve_gate() {
  local f off=ep2-off on=ep2-on i=0
  PROMPT_ARGS=()
  for f in "${BOUNDARY_FILES[@]}"; do PROMPT_ARGS+=(--prompt-file "$f"); done
  RESERVE_ARMS="$off $on"
  run_oracle_arm "$off" 2 "$DEVS" 2 0 0 "" 0
  run_oracle_arm "$on" 2 "$DEVS" 2 0 0 "" 1
  if ((MTP)); then
    RESERVE_ARMS+=" $off-mtp $on-mtp"
    run_oracle_arm "$off-mtp" 2 "$DEVS" 2 0 1 "" 0
    run_oracle_arm "$on-mtp" 2 "$DEVS" 2 0 1 "" 1
  fi
  if ((BASELINE_REPEAT)); then
    RESERVE_ARMS+=" $off-repeat"
    run_oracle_arm "$off-repeat" 2 "$DEVS" 2 0 0 "" 0
  fi

  : > "$OUT/reserve-eval.jsonl"
  reserve_eval_pair gate1-ar "$off" "$on" ar ""
  COMPARE_STATE=$PAIR_STATE
  if ((MTP)); then
    reserve_eval_pair gate1-mtp "$off-mtp" "$on-mtp" mtp ""
    MTP_STATE=$PAIR_STATE
  fi
  ((BASELINE_REPEAT == 0)) || reserve_eval_pair baseline-repeat "$off" "$off-repeat" ar "--baseline-repeat"
  for f in ${PERF_JSONS[@]+"${PERF_JSONS[@]}"}; do
    i=$((i + 1))
    reserve_eval_doc "perf-$i" perf "$f"
  done
  [[ -z $C1_MANIFEST ]] || reserve_eval_doc c1 c1 "$C1_MANIFEST"
  if ((RES_ANY_ERR)); then
    VERDICT=ERROR
  elif ((RES_ANY_FAIL)); then
    VERDICT=FAIL
  else
    VERDICT=PASS
  fi
}

# ------------------------------------------------------------------------ run
log "output directory: $OUT"
collect_env
select_cards
hash_model
build_all

EPARM=$MODE$N
if [[ $MODE == pp ]]; then
  run_pp_gate
elif [[ $MODE == reserve ]]; then
  run_reserve_gate
else
  run_oracle_arm "$REF_ARM" "$REF_RANKS" "$REF_DEVS" "$REF_RANKS"
  run_oracle_arm "$EPARM" "$N" "$DEVS" "$N"

  # (d) compare
  if oracle_arm_ok "$REF_ARM" && oracle_arm_ok "$EPARM"; then
    crc=0
    python3 "$COMPARE" "$OUT/oracle/$REF_ARM" "$OUT/oracle/$EPARM" --expect-ranks "$N" --ref-ranks "$REF_RANKS" \
      --json "$OUT/compare.json" > "$OUT/compare.txt" 2>&1 || crc=$?
    sed 's/^/    /' "$OUT/compare.txt"
    if ((crc == 0)); then
      VERDICT=PASS; COMPARE_STATE="PASS"
    elif ((crc == 1)); then
      VERDICT=FAIL; COMPARE_STATE="FAIL"
      if ((REF_RANKS == 1)); then REASON="EP$N output differs from the single card"; else REASON="EP$N output differs from the EP$REF_RANKS reference"; fi
    else
      VERDICT=ERROR; COMPARE_STATE="comparator error rc=$crc"; REASON="fn_ep_compare.py failed to run"
    fi
  else
    VERDICT=ERROR
    oracle_arm_ok "$REF_ARM" || COMPARE_STATE="$REF_ARM arm failed"
    oracle_arm_ok "$EPARM" || COMPARE_STATE="$EPARM arm failed (see oracle/$EPARM-receipt/stderr.txt)"
    REASON="$COMPARE_STATE; no comparison"
  fi

  # (e) speed cells, A/B interleaved per repetition
  if ((SKIP_SPEED)); then
    log "speed cells skipped"
  else
    EP_SPEED=1
    if ! oracle_arm_ok "$EPARM"; then
      EP_SPEED=0
      mkdir -p "$OUT/speed/$EPARM"
      echo "skipped: the $EPARM oracle arm failed to load/run" > "$OUT/speed/$EPARM/SKIPPED.txt"
    fi
    for ((rep = 1; rep <= REPS; rep++)); do
      run_bench_cell "$REF_ARM" "$REF_RANKS" "$REF_DEVS" decode "$rep" "$SHORT_PROMPT" 128
      ((EP_SPEED == 0)) || run_bench_cell "$EPARM" "$N" "$DEVS" decode "$rep" "$SHORT_PROMPT" 128
      run_bench_cell "$REF_ARM" "$REF_RANKS" "$REF_DEVS" prefill "$rep" "$LONG_PROMPT" 1
      ((EP_SPEED == 0)) || run_bench_cell "$EPARM" "$N" "$DEVS" prefill "$rep" "$LONG_PROMPT" 1
    done
  fi
fi

# (f) receipt
[[ -n $VERDICT ]] || VERDICT=ERROR
write_receipt
echo
cat "$OUT/summary.txt" || true
case $VERDICT in PASS) FINAL_RC=0 ;; FAIL) FINAL_RC=1 ;; *) FINAL_RC=1 ;; esac
exit "$FINAL_RC"
