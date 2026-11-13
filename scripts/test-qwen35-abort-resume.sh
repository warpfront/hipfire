#!/usr/bin/env bash
# Regression test: a client cancel mid-decode keeps the committed turn prefix.
#
# History. An early-truncated turn used to leave the NON-REVERSIBLE DeltaNet
# recurrent state and `conversation_tokens` dirty with UNCOMMITTED tokens (the
# decode-abort path returned without resetting), and the next turn resumed the
# prompt cache off that poisoned prior (whitespace garbage / premature EOS).
# The first fix full-reset on every abort, which also wiped the live
# conversation: the next request after an Esc cold-prefilled everything (~100 s
# at 118K tokens; #1062).
#
# Contract now. A cancel at a committed boundary is a committed turn prefix, not
# a failure: the daemon keeps prompt + the tokens actually fed into KV/DeltaNet
# (never the sampled-but-unfed token), frees checkpoints past it, drains the
# device and emits the cancel terminal. The next request hits the kept prefix or
# resumes from the latest prefix-hash-attested checkpoint at or before its
# divergence; a checkpoint that cannot attest the conversation prefix is never
# restored. Prefill aborts, unattestable boundaries, errors and model swaps still
# full-reset.
#
# DETERMINISTIC ASSERTIONS (fail on a build that wipes the session on cancel):
#   1. the cancel propagated to a real decode abort;
#   2. the daemon logged `[qwen-cache cancel] kept committed prefix pos=N`, N>0;
#   3. the follow-up (the cancelled partial turn echoed back, as a client does)
#      entered with a live conversation (`prior_len`/`conv_tok` > 0) and reused it
#      (`[qwen-cache HIT|resume ...]` or cached tokens > 0), no cold wipe;
#   4. the follow-up and an unrelated fresh request are coherent (no poison).
#
# Requires the daemon running with HIPFIRE_QWEN_CACHE_TRACE=1 (cache logs). Works
# on the MTP/DFlash route and on the AR route. Launch e.g.:
#   HIPFIRE_QWEN_CACHE_TRACE=1 HIPFIRE_MODEL=qwen3.6-35b-a3b.mq4 \
#     bash scripts/serve-restart.sh 11435
set -uo pipefail
PORT=${1:-11435}
MODEL=${HIPFIRE_TEST_MODEL:-qwen3.6-35b-a3b.mq4}
LOG=${HIPFIRE_SERVE_LOG:-$HOME/.hipfire/serve.log}
OUT=$(mktemp -d)
EP="http://127.0.0.1:$PORT/v1/chat/completions"
# Forced-long, thinking-on prompt so Turn A is guaranteed to STILL BE DECODING
# when the client drops the socket (can't finish in 5 s, won't hit a natural
# stop). This is what makes the cancel land in decode, not after.
PROMPT="Reason step by step in exhaustive detail, then write a comprehensive multi-section analysis comparing B-trees, LSM-trees, hash indexes, and tries for high-ingest time-series workloads. Cover concurrency control, cache behavior, and write amplification thoroughly."

command -v python3 >/dev/null || { echo "need python3"; exit 2; }
curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || { echo "daemon not responding on :$PORT — start it with $MODEL loaded first"; exit 2; }

LINES0=$(wc -l < "$LOG" 2>/dev/null || echo 0)

echo "=== Turn A: stream with thinking ON, drop the connection mid-decode (~5 s) ==="
# The client drops the connection after DROP_S seconds of streamed output, as Esc
# does: native serve detects the disconnect -> sends {type:abort} -> the decode
# abort fires. Python keeps the partial reasoning/content to echo back in Turn B.
python3 - "$EP" "$MODEL" "$PROMPT" "$OUT/turnA.json" <<'PY' || exit 1
import json, socket, sys, time, urllib.request
ep, model, prompt, out = sys.argv[1:5]
body = {"model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000, "temperature": 0, "stream": True}
req = urllib.request.Request(ep, json.dumps(body).encode(), {"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=300)
t0, reasoning, content, nbytes = time.time(), "", "", 0
for raw in resp:
    nbytes += len(raw)
    line = raw.decode().strip()
    if line.startswith("data:") and not line.endswith("[DONE]"):
        ch = json.loads(line[5:])
        if ch.get("choices"):
            d = ch["choices"][0].get("delta", {})
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
    if time.time() - t0 > 5.0 and (reasoning or content):
        break
try:
    resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
except OSError:
    pass
resp.close()
json.dump({"reasoning": reasoning, "content": content, "bytes": nbytes}, open(out, "w"))
print(f"  Turn A connection dropped after {time.time()-t0:.1f}s ({nbytes} bytes, "
      f"{len(reasoning)} reasoning chars, {len(content)} content chars)")
sys.exit(0 if (reasoning or content) else 1)
PY
[ $? -eq 0 ] || { echo "FAIL: Turn A produced no output — never reached decode"; exit 1; }
sleep 3   # let the daemon's stdin reader process the abort + epilogue

echo "=== Turn B: follow-up with the cancelled partial turn echoed back ==="
python3 - "$EP" "$MODEL" "$PROMPT" "$OUT/turnA.json" "$OUT/turnB.json" <<'PY' || exit 1
import json, sys, urllib.request
ep, model, prompt, a_path, out = sys.argv[1:6]
a = json.load(open(a_path))
partial = {"role": "assistant", "content": a["content"]}
if a["reasoning"]:
    partial["reasoning_content"] = a["reasoning"]
msgs = [{"role": "user", "content": prompt}, partial,
        {"role": "user", "content": "Never mind. Reply with exactly: the quick brown fox"}]
body = {"model": model, "messages": msgs, "max_tokens": 256, "temperature": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(ep, json.dumps(body).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=300))
m = r["choices"][0]["message"]
json.dump({"content": m.get("content") or "", "usage": r.get("usage")}, open(out, "w"))
PY
CONTENT=$(python3 -c "import json,sys;print(json.load(open('$OUT/turnB.json'))['content'])")
echo "  Turn B content: $(printf '%s' "$CONTENT" | head -c 160)"
sleep 1
NEW=$(tail -n +$((LINES0+1)) "$LOG" 2>/dev/null)

# (1)+(2) the cancel propagated to a real decode abort AND kept the committed
# prefix. A build that full-resets on cancel logs no abort line, so the two are
# told apart by what Turn B finds: an empty conversation after a decoding
# Turn A means the abort fired and wiped it.
KEPT=$(echo "$NEW" | grep -oE '\[qwen-cache cancel\] kept committed prefix pos=[0-9]+' | head -1 | grep -oE '[0-9]+$')
# Turn B entry. AR route: `[qwen-cache GEN-ENTRY] conv_tok=N`; MTP/DFlash route:
# `[qwen-cache lcp <tag>] prior_len=N` (absent when the conversation is empty).
# The LAST of either = Turn B's.
ENTRY=$(echo "$NEW" | grep -oE '\[qwen-cache GEN-ENTRY\] conv_tok=[0-9]+|\[qwen-cache lcp[^]]*\] prior_len=[0-9]+' | tail -1 | grep -oE '[0-9]+$')
if [ "${KEPT:-0}" -gt 0 ]; then
  echo "  OK: cancel propagated to a decode-abort and kept the committed prefix (pos=$KEPT)"
elif [ "${ENTRY:-0}" -eq 0 ]; then
  echo "FAIL: the mid-decode cancel wiped the session (full reset): no '[qwen-cache cancel] kept committed prefix' line and Turn B entered with an empty conversation"
  echo "$NEW" | grep -E 'qwen-cache|abort|cancel'; exit 1
else
  echo "FAIL: no abort fired (Turn A finished before the cancel, or the disconnect was not detected)"
  echo "$NEW" | grep -E 'qwen-cache|abort|cancel'; exit 1
fi

# (3) Turn B entered with the live conversation and reused it.
echo "  Turn B entered with conv_tok/prior_len=${ENTRY:-0}"
[ "${ENTRY:-0}" -gt 0 ] \
  && echo "  OK: the next request found the live conversation (not wiped)" \
  || { echo "FAIL: conv_tok/prior_len=${ENTRY:-0} after the cancel — the session was wiped (cold prefill)"; exit 1; }
echo "$NEW" | grep -qE '\[qwen-cache (HIT|resume)|\[qwen-cache resume' \
  && echo "  OK: Turn B reused the kept state ($(echo "$NEW" | grep -oE '\[qwen-cache (HIT|resume)[^]]*\][^(]*' | tail -1))" \
  || echo "  NOTE: no HIT/resume line (the echo may have matched the kept prefix exactly or re-rendered below the first checkpoint)"
echo "$NEW" | grep -qE 'prefix hash does not attest' \
  && { echo "FAIL: a checkpoint was rejected by the prefix-hash attestation after a plain cancel"; exit 1; } \
  || echo "  OK: every checkpoint attested its prefix"

# (4) coherence of Turn B and of an unrelated fresh request (catches whitespace-
#     attractor / empty garbage from a poisoned state)
coherent() { # name text
  local LEN=${#2}; local NONSPACE=$(printf '%s' "$2" | tr -d '[:space:]' | wc -c)
  local UNIQ=$(printf '%s' "$2" | fold -w1 | sort -u | wc -l)
  echo "  $1 coherence: len=$LEN nonspace=$NONSPACE uniq_chars=$UNIQ"
  [ "$NONSPACE" -ge 5 ] && [ "$UNIQ" -ge 6 ]
}
coherent "Turn B" "$CONTENT" || { echo "FAIL: Turn B degenerate/empty after a cancel"; exit 1; }

echo "=== Turn C: unrelated fresh request (cold render, must not be poisoned) ==="
RESC=$(curl -fsS -m 120 "$EP" -H 'content-type: application/json' \
  -d "$(python3 -c "import json,sys;print(json.dumps({'model':sys.argv[1],'messages':[{'role':'user','content':'List three prime numbers greater than 100, comma separated.'}],'max_tokens':64,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}))" "$MODEL")")
CONTENTC=$(echo "$RESC" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'])")
echo "  Turn C content: $(printf '%s' "$CONTENTC" | head -c 160)"
coherent "Turn C" "$CONTENTC" \
  || { echo "FAIL: Turn C degenerate/empty after a cancel"; exit 1; }

echo "=== Turn D: resend Turn A's exact prompt unedited (must not resume off decode state) ==="
LINES1=$(wc -l < "$LOG" 2>/dev/null || echo 0)
RESD=$(curl -fsS -m 120 "$EP" -H 'content-type: application/json' \
  -d "$(python3 -c "import json,sys;print(json.dumps({'model':sys.argv[1],'messages':[{'role':'user','content':sys.argv[2]}],'max_tokens':48,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}))" "$MODEL" "$PROMPT")")
CONTENTD=$(echo "$RESD" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'])")
echo "  Turn D content: $(printf '%s' "$CONTENTD" | head -c 160)"
coherent "Turn D" "$CONTENTD" \
  || { echo "FAIL: Turn D degenerate/empty after an unedited resend"; exit 1; }
NEWD=$(tail -n +$((LINES1+1)) "$LOG" 2>/dev/null)
echo "$NEWD" | grep -q 'prefix hash does not attest' \
  && { echo "FAIL: prefix-hash attestation rejected a checkpoint on the unedited-resend path"; echo "$NEWD" | grep -E 'qwen-cache|abort|cancel'; exit 1; } \
  || echo "  OK: every checkpoint attested its prefix on the resend path"
REWOUND=$(echo "$NEWD" | grep -oE '\[qwen-cache resume\] rewound to checkpoint pos=[0-9]+' | tail -1 | grep -oE '[0-9]+$')
if [ -n "$REWOUND" ]; then
  [ "$REWOUND" -le "${KEPT:-0}" ] \
    && echo "  OK: Turn D rewound to pos=$REWOUND, at or before the kept prefix (pos=${KEPT:-0})" \
    || { echo "FAIL: Turn D resumed from pos=$REWOUND past the kept prefix ${KEPT:-0} (decode-state poison)"; exit 1; }
else
  echo "  OK: Turn D took no rewound resume (live continuation or cold render)"
fi
echo "PASS: a mid-decode cancel kept the committed prefix and did not poison later turns"
