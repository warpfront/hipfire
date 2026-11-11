#!/usr/bin/env python3
"""Extract byte-exact CPU G1 samples (W3 cold qt131/132 + MQ4 hot qt44/53) from a SHA-pinned mq3-xts HFQ artifact.

Usage: python3 tools/mq3_k2_extract_samples.py --artifact qwen3.8-flash-next-base.mq3-xts \
    --expect-sha256 HEX --output-dir DIR [--layers 0,23,47] [--cold 3] [--hot 2]

Writes `sample.json` + raw `L{layer}.E{expert}.{kind}.bin` in the format `qwen4_moe_sym_gfx1100_g1/samples.rs` reads
(`{seed, layers, provisional:false, artifact_path, artifact_sha256, samples:[{layer, expert, hot, tensors:[{kind, qt,
shape, group, file, sha256, bytes, artifact_offset}]}]}`), usable for `--codec w3` per-codec runs and the mixed
`--mixed-codeobject mq4=... --mixed-codeobject w3=...` run. The HFQ index parser and hashing are those of
`e8p_k1_extract_samples.py` (imported, same directory). The destination must not exist; `sample.json` is written last.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e8p_k1_extract_samples import CHUNK, NAME, hash_range, parse, require  # noqa: E402

# kind -> (cold qt, hot qt, shape, cold bytes, hot bytes, hot group)
KINDS = {
    "gate_up_proj": (131, 44, [1280, 2560], 1_280_000, 1_740_800, 256),
    "down_proj": (132, 53, [2560, 640], 640_000, 870_400, 128),
}


def select(entries, layers, n_cold, n_hot):
    table = {layer: {} for layer in layers}
    for entry in entries:
        match = NAME.fullmatch(entry["name"])
        if not match:
            continue
        layer, expert = map(int, match.group(1, 2))
        if layer in table:
            table[layer].setdefault(expert, {})[match.group(3)] = entry
    picked = []
    for layer in layers:
        cold, hot = [], []
        for expert, proj in sorted(table[layer].items()):
            if set(proj) != set(KINDS):
                continue
            qts = {kind: proj[kind]["qt"] for kind in KINDS}
            if all(qts[k] == KINDS[k][0] for k in KINDS):
                cold.append(expert)
            elif all(qts[k] == KINDS[k][1] for k in KINDS):
                hot.append(expert)
        require(len(cold) >= n_cold and len(hot) >= n_hot,
                "layer %d needs %d cold W3 and %d hot MQ4 experts, has %d/%d" % (layer, n_cold, n_hot, len(cold), len(hot)))
        # first, last and evenly spread cold experts; first and last hot experts.
        spread = [cold[round(i * (len(cold) - 1) / max(n_cold - 1, 1))] for i in range(n_cold)]
        for expert in dict.fromkeys(spread):
            picked.append((layer, expert, False, table[layer][expert]))
        for expert in dict.fromkeys([hot[0], hot[-1]][:n_hot]):
            picked.append((layer, expert, True, table[layer][expert]))
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", required=True, type=Path)
    p.add_argument("--expect-sha256", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--layers", default="0,23,47")
    p.add_argument("--cold", type=int, default=3)
    p.add_argument("--hot", type=int, default=2)
    args = p.parse_args()
    require(re.fullmatch(r"[0-9a-fA-F]{64}", args.expect_sha256), "expected SHA256 must be 64 hex digits")
    layers = [int(x) for x in args.layers.split(",")]
    artifact = args.artifact.resolve(strict=True)
    with artifact.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        sha = hash_range(stream, 0, size)
        require(sha == args.expect_sha256.lower(), "artifact SHA256 mismatch: " + sha)
        picked = select(parse(stream, size), layers, args.cold, args.hot)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        samples = []
        for layer, expert, hot, projections in picked:
            tensors = []
            for kind, (cold_qt, hot_qt, shape, cold_bytes, hot_bytes, hot_group) in KINDS.items():
                entry = projections[kind]
                want_qt, want_bytes, group = (hot_qt, hot_bytes, hot_group) if hot else (cold_qt, cold_bytes, 128)
                require((entry["qt"], entry["shape"], entry["bytes"]) == (want_qt, shape, want_bytes),
                        "unexpected qtype/shape/length: " + entry["name"])
                name = "L%d.E%d.%s.bin" % (layer, expert, kind)
                stream.seek(entry["artifact_offset"])
                remaining, digest = entry["bytes"], hashlib.sha256()
                with (args.output_dir / name).open("xb") as out:
                    while remaining:
                        block = stream.read(min(CHUNK, remaining))
                        require(block, "unexpected EOF")
                        out.write(block)
                        digest.update(block)
                        remaining -= len(block)
                require(digest.hexdigest() == hash_range(stream, entry["artifact_offset"], entry["bytes"]),
                        "reread mismatch: " + name)
                tensors.append(dict(kind=kind, qt=want_qt, shape=shape, group=group, file=name,
                                    sha256=digest.hexdigest(), bytes=entry["bytes"],
                                    artifact_offset=entry["artifact_offset"]))
            samples.append(dict(layer=layer, expert=expert, hot=hot, tensors=tensors))
    manifest = dict(seed=0, layers=layers, provisional=False, artifact_path=str(artifact), artifact_sha256=sha,
                    qtypes={"cold": {k: v[0] for k, v in KINDS.items()}, "hot": {k: v[1] for k, v in KINDS.items()}},
                    extractor=str(Path(__file__).resolve()), samples=samples)
    with (args.output_dir / "sample.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(args.output_dir / "sample.json")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print("mq3 sample extraction failed: %s" % error, file=sys.stderr)
        sys.exit(1)
