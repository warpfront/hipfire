#!/usr/bin/env bash
# Developer-only evaluation harness for HIPFIRE_DFLASH_ONLINE_TUNE (not CI, not acceptance).
# Pause other GPU users before running; set LD_LIBRARY_PATH to your ROCm if needed.
# Caches (dumps, shipping arm, q8 HIPFIRE_HOME) live under ${DFLASH_TUNE_CACHE:-~/.cache/hipfire-online}.
# Online DFlash draft tuning (halo, gfx1151), two target/draft pairs, greedy:
#   q9: Qwen3.5-9B MQ4 + qwen35-9b-dflash-mq4 (qwen35 chain DFlash, strong draft)
#   q8: Qwen3-8B MQ4 + qwen3-8b-dflash        (generic llama-family DFlash, weak draft)
#
# Primary: online_tokwin_ratio = geomean over 10 committed long prompts x 2 pairs of
#   (tokens per verify window, tuned) / (tokens per verify window, shipping DFlash). One cold
#   request per fresh daemon, as a server sees a new conversation. Greedy decode is deterministic
#   per policy, so this is exact for a given build (no thermal noise); per-prompt values still
#   move when a policy change re-rolls batched-verify near-ties, hence 20 prompts.
#   The shipping arm (tuner unset) is cached per pair+prompt+max_tokens (it never runs tuner
#   code); delete $C/shipping if non-tuner engine code changes.
# Why not replay as primary: fixed-start replay of argmax sessions cannot see that longer
#   acceptance moves later block starts onto harder positions; it overstated two changes
#   (+5% replay, -1% online over 20 prompts). Replay stays a diagnostic.
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
declare -A DUMPS=([q9]=$C/v3 [q8]=$C/q8)
declare -A MAXTOK=([q9]=1536 [q8]=1024)


out=$(mktemp -d)
bench() { # pair outfile-stem prompt-file mode(stats|on|off) [dump]
  local tune=(); [ "$4" = off ] || tune=(HIPFIRE_DFLASH_ONLINE_TUNE="$4")
  env ${HOMEDIR[$1]:+HIPFIRE_HOME=${HOMEDIR[$1]}} "${tune[@]}" ${5:+HIPFIRE_DFLASH_ONLINE_DUMP=$5} \
    timeout --foreground 900 "$HF" bench "${MODEL[$1]}" --spec dflash --runs 1 --warmups 0 \
    --max-tokens "${MAXTOK[$1]}" --backend noslots --workload stateless \
    --prompt-file "$3" --json >"$2.json" 2>"$2.err" \
    || { tail -30 "$2.err" >&2; echo "bench $1 $2 FAILED" >&2; exit 1; }
}

for pair in q9 q8; do
  mkdir -p "${DUMPS[$pair]}" "$C/shipping/$pair"
  dumps=()
  for f in "$P"/*.txt; do
    n=$(basename "$f" .txt); key="$n-$(md5sum <"$f" | cut -c1-12)-${MAXTOK[$pair]}"
    d="${DUMPS[$pair]}/$n-$(md5sum <"$f" | cut -c1-12).bin"
    [ -s "$d" ] || bench "$pair" "$out/dump" "$f" stats "$d"
    dumps+=("$d")
    [ -s "$C/shipping/$pair/$key.err" ] || bench "$pair" "$C/shipping/$pair/$key" "$f" off
    bench "$pair" "$out/$pair.$n" "$f" on
    cp "$C/shipping/$pair/$key.err" "$out/$pair.$n.A.err"; cp "$C/shipping/$pair/$key.json" "$out/$pair.$n.A.json"
  done
  "$REPLAY" --carry-loo "${dumps[@]}" >"$out/$pair.replay.txt"
done

python3 - "$out" <<'EOF'
import json, math, os, re, sys
d = sys.argv[1]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
def run(stem):
    err = open(os.path.join(d, stem + ".err")).read()
    toks, wins = map(int, re.findall(r"decode \((\d+) tok, (\d+) windows", err)[-1])
    return toks / wins, json.load(open(os.path.join(d, stem + ".json")))["decode_tok_s"]["median"]
allw, alls = [], []
for pair in ("q9", "q8"):
    w, s = [], []
    for f in sorted(os.listdir(d)):
        m = re.match(pair + r"\.([a-z_]+)\.err$", f)
        if not m:
            continue
        n = m.group(1)
        b, a = run(f"{pair}.{n}"), run(f"{pair}.{n}.A")
        w.append(b[0] / a[0]); s.append(b[1] / a[1])
        print("METRIC %s_%s_tokwin=%.4f" % (pair, n, b[0] / a[0]))
    print("METRIC %s_tokwin_ratio=%.4f" % (pair, g(w)))
    print("METRIC %s_tok_s_ratio=%.4f" % (pair, g(s)))
    rep = open(os.path.join(d, pair + ".replay.txt")).read()
    print("METRIC %s_replay=%s" % (pair, re.search(r"sessions: ([0-9.]+)", rep).group(1)))
    allw += w; alls += s
print("METRIC online_tokwin_ratio=%.4f" % g(allw))
print("METRIC online_tok_s_ratio=%.4f" % g(alls))
EOF
rm -rf "$out"
