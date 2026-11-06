#!/usr/bin/env python3
"""Prefill-chunk helpers of scripts/fn-ep-gate.sh PP mode.

A PP exactness pair is only byte-comparable when both arms prefill the same chunk (the chunk
changes numerics), but every PP load negotiates its own canonical chunk from its cards' free VRAM.
The gate therefore runs the reference arm first, reads the chunk it settled on from its log, and
pins that chunk on every later arm (HIPFIRE_PREFILL_CHUNK_ROWS); an explicit value in EXTRA_ENV wins.

    fn_ep_gate_chunk.py user  "<EXTRA_ENV>"      -> prints N when EXTRA_ENV sets HIPFIRE_PREFILL_CHUNK_ROWS=N
    fn_ep_gate_chunk.py parse <arm stderr file>  -> prints "N<TAB>source" for the chunk that load used
Exit 1 (nothing printed) when there is no such value.
"""
import re
import sys

PP_LINE = re.compile(r"qwen4 PP canonical chunk: (\d+) rows")
SINGLE_LINE = re.compile(r"qwen4 prefill chunk: (\d+) rows")
USER_PIN = re.compile(r"(?:^|\s)(?:export\s+)?HIPFIRE_PREFILL_CHUNK_ROWS=(\d+)(?=\s|$)")


def user_chunk(extra_env):
    """The last explicit HIPFIRE_PREFILL_CHUNK_ROWS in an EXTRA_ENV string (rounded down to whole
    256-row tiles as the loader does; a value below one tile is no pin)."""
    found = USER_PIN.findall(extra_env or "")
    if not found:
        return None
    rows = int(found[-1]) // 256 * 256
    return rows if rows >= 256 else None


def load_chunk(stderr_text):
    """(rows, source) of the chunk a load settled on: the PP mesh's negotiated canonical chunk, else
    the single-card load's own `qwen4 prefill chunk` line. The last line wins."""
    pp = PP_LINE.findall(stderr_text)
    if pp:
        return int(pp[-1]), "PP canonical chunk line"
    single = SINGLE_LINE.findall(stderr_text)
    if single:
        return int(single[-1]), "single-card prefill chunk line"
    return None


def main(argv):
    if len(argv) == 3 and argv[1] == "user":
        rows = user_chunk(argv[2])
        if rows is None:
            return 1
        print(rows)
        return 0
    if len(argv) == 3 and argv[1] == "parse":
        try:
            with open(argv[2], encoding="utf-8", errors="replace") as handle:
                found = load_chunk(handle.read())
        except OSError:
            return 1
        if found is None:
            return 1
        print("%d\t%s" % found)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
