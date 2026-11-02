#!/usr/bin/env python3
"""Long-context multi-needle NIAH fixtures past the native 262,144-token window (YarnQwen).

Reuses the repo's committed multi-needle format (benchmarks/longctx/niah/generate_niah_multi.py: same
filler words, needle sentence shape, record keys) so `scripts/serve_harness.py --niah-file` consumes it
unchanged. Needles sit both inside and past the native window; the filler length is calibrated against the
Qwen3.x/Flash-Next tokenizer so the measured token count lands on target.

  gen_niah_long.py TOKENIZER_JSON OUT_DIR
writes niah_long_{300k,500k}.jsonl + manifest.json (token counts, needle token offsets, sha256).
"""
import hashlib, json, random, sys
from pathlib import Path
from tokenizers import Tokenizer

WORDS = (
    "the quick brown fox jumps over the lazy dog and then settles down for a "
    "long restful nap on a warm patch of sunlit grass while a gentle breeze "
    "carries the scent of distant pine forests across the meadow where small "
    "wildflowers nod their heads in agreement with the rhythm of the afternoon "
    "and time itself seems to pause for just a moment so that everything alive "
    "can breathe in synchrony with the slow steady pulse of the surrounding "
    "world before the sun begins its descent toward the western horizon "
    "painting clouds in shades of amber rose and indigo as evening approaches"
).split()
NEEDLES = [
    ("courier color", "indigo-octahedron-9931"),
    ("vault key", "fenrir-quartz-2247"),
    ("safehouse city", "saint-petersburg-rotunda-5808"),
    ("launch phrase", "cobalt-marmoset-6120"),
]
QUESTION = (
    "Based on the document above, list each of the following: the courier color secret, the vault key "
    "secret, the safehouse city secret, and the launch phrase secret. Use one short sentence per fact."
)
NATIVE = 262_144
# (name, target tokens, needle depths): the long fixtures have needles inside and past the native window;
# 48k is the in-window control that the standing Halo rule (<=57K without Main's OK) allows.
FIXTURES = [("48k", 48_000, [0.10, 0.40, 0.70, 0.95]), ("300k", 300_000, [0.10, 0.50, 0.90, 0.97]),
            ("500k", 500_000, [0.10, 0.45, 0.70, 0.95])]
SEED = 0x5407_FEEF


def filler(rng, chars):
    out, n = [], 0
    while n < chars:
        w = WORDS[rng.randrange(len(WORDS))]
        out.append(w)
        n += len(w) + 1
    return " ".join(out)


def assemble(target, depths, cpt):
    rng = random.Random(SEED + target)
    frags, prev = [], 0.0
    for depth, (key, value) in zip(depths, NEEDLES):
        frags.append(filler(rng, int(target * (depth - prev) * cpt)))
        frags.append(f"\n\nThe {key} is {value}.\n\n")
        prev = depth
    frags.append(filler(rng, int(target * (1.0 - prev) * cpt)))
    return "".join(frags)


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    manifest = {"tokenizer_sha256": hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(), "fixtures": []}
    for name, target, depths in FIXTURES:
        cpt = 3.5
        for _ in range(6):
            text = assemble(target, depths, cpt)
            n = len(tok.encode(text, add_special_tokens=False).ids)
            if abs(n - target) <= target * 0.002:
                break
            cpt *= target / n
        ids = tok.encode(text, add_special_tokens=False)
        offsets = []
        for key, value in NEEDLES:
            assert text.count(value) == 1, value
            char = text.index(value)
            offsets.append(next(i for i, (s, e) in enumerate(ids.offsets) if e > char))
        record = {
            "context_tokens": len(ids.ids),
            "needles": [{"key": k, "value": v, "depth": d, "token_offset": o}
                        for d, (k, v), o in zip(depths, NEEDLES, offsets)],
            "question": QUESTION,
            "expected_answer_substrings": [v for _, v in NEEDLES],
            "min_recovered": 3,
            "filler_text": text,
        }
        path = out / f"niah_long_{name}.jsonl"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["fixtures"].append({
            "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "filler_tokens": len(ids.ids), "chars_per_token": round(len(text) / len(ids.ids), 4),
            "needle_token_offsets": offsets,
            "needles_past_native": sum(o >= NATIVE for o in offsets),
        })
        print(manifest["fixtures"][-1])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
