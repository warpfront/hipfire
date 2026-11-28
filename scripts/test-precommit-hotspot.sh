#!/usr/bin/env bash
# Routing test for the HOTSPOT early exit in .githooks/pre-commit (HIPF-630).
# Runs the real hook in a scratch repo. Gate scripts are stubs that record calls.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cd "$T" && git init -q . && mkdir -p .githooks scripts
cp "$ROOT/.githooks/pre-commit" .githooks/
for g in "$ROOT"/scripts/*gate*.sh "$ROOT"/scripts/verify-bind-thread.sh "$ROOT"/scripts/check-fmt-bomb.sh; do
    [ -e "$g" ] && case "$g" in *gate*) b="echo \"\$0\" >> \"$T/calls\"";; *) b=:;; esac && printf "#!/bin/sh\n%s\n" "$b" > "scripts/$(basename "$g")" \
        && chmod +x "scripts/$(basename "$g")"
done
printf "pass\n" > scripts/check-crate-maps.py; printf ":\n" > scripts/leanup-ratchets.sh
fail=0
run() { # name expect(gates|none) paths...
    local name=$1 expect=$2; shift 2
    : > "$T/calls"; git reset -q; mkdir -p benchmarks/engine-loop-ab crates/x/src
    for f in "$@"; do mkdir -p "$(dirname "$f")"; echo "$RANDOM" > "$f"; git add -f "$f"; done
    SKIP_REGEN=1 SKIP_FMT_CHECK=1 HIPFIRE_SKIP_AGENTIC_GATE=0 bash .githooks/pre-commit >"$T/out" 2>&1
    if [ "$expect" = none ] && [ -s "$T/calls" ]; then echo "FAIL $name: gates ran"; fail=1
    elif [ "$expect" = gates ] && ! grep -q gate "$T/calls"; then echo "FAIL $name: no gate"; fail=1
    else echo "ok   $name"; fi
}
run json-only none benchmarks/engine-loop-ab/qwen4-mq4xts-mtp.lane-tau.json
run json-nonascii none "benchmarks/engine-loop-ab/mq4-café.json"
run code gates crates/x/src/forward.rs
run mixed gates crates/x/src/forward.rs benchmarks/engine-loop-ab/mq4-a.json
exit $fail
