#!/usr/bin/env bash
# Developer-only evaluation harness for HIPFIRE_DFLASH_ONLINE_TUNE (not CI, not acceptance).
# Pause other GPU users before running; set LD_LIBRARY_PATH to your ROCm if needed.
# Caches (dumps, shipping arm, q8 HIPFIRE_HOME) live under ${DFLASH_TUNE_CACHE:-~/.cache/hipfire-online}.
# Online DFlash draft tuning (halo, gfx1151), two target/draft pairs, greedy:
#   q9: Qwen3.5-9B MQ4 + qwen35-9b-dflash-mq4 (qwen35 chain DFlash, strong draft)
#   q8: Qwen3-8B MQ4 + qwen3-8b-dflash        (generic llama-family DFlash, weak draft)
#
# Primary: sim_tokwin_ratio = geomean over both pairs of the per-pair geomean (10 committed long
#   prompts, benchmarks/prompts/online_tune/) of tokens per verify window, tuned policy vs argmax,
#   simulated over SWEEP dumps (HIPFIRE_DFLASH_ONLINE_TUNE=sweep: the draft's top-K recorded at
#   every position of the greedy text). `dflash_online_replay --simulate` walks each policy with
#   its OWN block starts (accept the longest matching prefix, start after the bonus), one cold
#   request per session -- what a server does, minus batched-verify near-tie text flips.
#   Validated: simulated argmax windows == shipping windows (q9 code_edit 80 == 80, all sessions
#   within a few %); baseline policy sim q9 x1.024 / q8 x1.088 vs online x1.036 / x1.095;
#   per-prompt online-vs-sim residual SD 2.8%. Both arms share one text, so the ratio carries no
#   text-divergence noise (online A/B: ~+-1% per 10-prompt pair).
# Fixed-start replay of argmax sessions is NOT used: it ignores that a policy accepting more
#   moves later block starts onto harder positions (overstated two changes by ~5%).
set -euo pipefail
cd "$(dirname "$0")/../.."
unset HIPFIRE_DFLASH_ONLINE_TUNE HIPFIRE_DFLASH_ONLINE_HP HIPFIRE_DFLASH_ONLINE_DUMP HIPFIRE_HOME

cargo build -q --release 2>/dev/null
cargo build -q --release -p hipfire-runtime --example dflash_online_replay 2>/dev/null
T=${CARGO_TARGET_DIR:-target}
HF=$T/release/hipfire
REPLAY=$T/release/examples/dflash_online_replay
P=benchmarks/prompts/online_tune
C=${DFLASH_TUNE_CACHE:-$HOME/.cache/hipfire-online}

# q8 needs a non-registry draft: an isolated HIPFIRE_HOME (links to ~/.hipfire) whose
# config sets developer.dflash_draft and speculation.dflash = "on".
Q8HOME=$C/hf8home
if [ ! -f "$Q8HOME/config.toml" ]; then
  mkdir -p "$Q8HOME"
  for e in "$HOME"/.hipfire/*; do [ "$(basename "$e")" = config.toml ] || ln -sfn "$e" "$Q8HOME/$(basename "$e")"; done
  { sed 's/^dflash = "off"/dflash = "on"/' "$HOME/.hipfire/config.toml"
    printf '\n[developer]\ndflash_draft = "%s"\n' "$HOME/.hipfire/models/qwen3-8b-dflash.hfq"; } >"$Q8HOME/config.toml"
fi
declare -A MODEL=([q9]=$HOME/.hipfire/models/qwen3.5-9b.mq4 [q8]=$HOME/.hipfire/models/qwen3-8b.mq4)
declare -A HOMEDIR=([q9]="" [q8]=$Q8HOME)
declare -A MAXTOK=([q9]=1536 [q8]=1024)

out=$(mktemp -d)
for pair in q9 q8; do
  mkdir -p "$C/sweep/$pair"
  dumps=()
  for f in "$P"/*.txt; do
    d="$C/sweep/$pair/$(basename "$f" .txt).bin"
    if [ ! -s "$d" ]; then   # one-time, ~2 min per prompt
      env ${HOMEDIR[$pair]:+HIPFIRE_HOME=${HOMEDIR[$pair]}} HIPFIRE_DFLASH_ONLINE_TUNE=sweep \
        HIPFIRE_DFLASH_ONLINE_DUMP="$d" timeout --foreground 1800 "$HF" bench "${MODEL[$pair]}" \
        --spec dflash --runs 1 --warmups 0 --max-tokens "${MAXTOK[$pair]}" --backend noslots \
        --workload stateless --prompt-file "$f" --json >"$out/sw.json" 2>"$out/sw.err" \
        || { tail -30 "$out/sw.err" >&2; exit 1; }
    fi
    dumps+=("$d")
  done
  "$REPLAY" --simulate "${dumps[@]}" >"$out/$pair.sim.txt"
  cat "$out/$pair.sim.txt" >&2
done

python3 - "$out" <<'EOF'
import math, os, re, sys
d = sys.argv[1]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
pairs = []
for pair in ("q9", "q8"):
    t = open(os.path.join(d, pair + ".sim.txt")).read()
    r = float(re.search(r"SIM GEOMEAN tuned/argmax tok/win over \d+ sessions: ([0-9.]+)", t).group(1))
    pairs.append(r)
    print("METRIC %s_sim=%.4f" % (pair, r))
    for n, a, b in re.findall(r"/([a-z_]+)\.bin: argmax tok/win=([0-9.]+) .*? tuned tok/win=([0-9.]+)", t):
        print("METRIC %s_sim_%s=%.4f" % (pair, n, float(b) / float(a)))
print("METRIC sim_tokwin_ratio=%.4f" % g(pairs))
EOF
rm -rf "$out"
