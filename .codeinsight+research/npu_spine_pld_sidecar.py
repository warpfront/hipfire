#!/usr/bin/env python3
import json
import sys


def pld(tokens, max_tokens):
    for n in (5, 4, 3):
        if len(tokens) <= n:
            continue
        suffix_start = len(tokens) - n
        suffix = tokens[suffix_start:]
        haystack = tokens[:suffix_start]
        found = -1
        for i in range(len(haystack) - n, -1, -1):
            if haystack[i : i + n] == suffix:
                found = i
                break
        if found < 0:
            continue
        cont_start = found + n
        cont_end = min(cont_start + max_tokens, suffix_start)
        out = tokens[cont_start:cont_end]
        if len(out) >= 3:
            return out
    return []


def main():
    req = json.load(sys.stdin)
    max_tokens = int(req.get("max_tokens") or 8)
    if req.get("pld_tokens"):
        tokens = req["pld_tokens"][:max_tokens]
        source = "pld-pass"
        confidence = 0.95
    else:
        tokens = pld(req.get("tail_tokens") or [], max_tokens)
        source = "sidecar-pld"
        confidence = 0.80 if tokens else 0.0
    print(json.dumps({"tokens": tokens, "confidence": confidence, "source": source}))


if __name__ == "__main__":
    main()
