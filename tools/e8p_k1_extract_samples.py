#!/usr/bin/env python3
"""Extract byte-exact CPU E8P G1 samples from a SHA-pinned HFQ artifact.

Usage: python3 tools/e8p_k1_extract_samples.py --artifact MODEL.hfq \
    --expect-sha256 HEX --output-dir DIR

HFQ v1 metadata/index layout follows verify_slab_artifact.py's parser:
JSON metadata, u32 count, then length-prefixed names and tensor descriptors;
payload offsets accumulate from the header's data offset. No release imports.
Qtypes 135/136 identify cold experts. E8P payloads contain planar little-endian
u16 words (one per eight weights), followed by f16 scales (one per group128).
The destination must not exist, so a failed run cannot leave a stale success
manifest. sample.json is published only after all reread checks succeed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

CHUNK = 8 << 20
KINDS = {"gate_up_proj": (135, [1280, 2560]),
         "down_proj": (136, [2560, 640])}
NAME = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_up_proj|down_proj)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_range(stream, offset, size):
    stream.seek(offset)
    digest = hashlib.sha256()
    while size:
        block = stream.read(min(CHUNK, size))
        require(block, "unexpected EOF while hashing artifact range")
        digest.update(block)
        size -= len(block)
    return digest.hexdigest()


def parse(stream, file_size):
    stream.seek(0)
    header = stream.read(32)
    require(len(header) == 32, "truncated HFQ header")
    magic, version, _arch, count, metadata_offset, data_offset = struct.unpack("<4sIIIQQ", header)
    require(magic == b"HFQM" and version == 1, "expected HFQM version 1")
    require(32 <= metadata_offset <= data_offset <= file_size, "invalid HFQ metadata/data offsets")
    stream.seek(metadata_offset)
    blob = stream.read(data_offset - metadata_offset)
    require(len(blob) == data_offset - metadata_offset, "truncated HFQ index")
    text = blob.decode("utf-8", "surrogateescape")
    _metadata, end = json.JSONDecoder().raw_decode(text)
    off = len(text[:end].encode("utf-8", "surrogateescape"))

    def unpack(fmt):
        nonlocal off
        size = struct.calcsize(fmt)
        require(off + size <= len(blob), "truncated HFQ index descriptor")
        result = struct.unpack_from(fmt, blob, off)
        off += size
        return result

    indexed_count, = unpack("<I")
    require(indexed_count == count, "HFQ header/index tensor count mismatch")
    entries = []
    position = data_offset
    names = set()
    for _ in range(count):
        length, = unpack("<H")
        require(off + length <= len(blob), "truncated HFQ tensor name")
        name = blob[off:off + length].decode("utf-8")
        off += length
        require(name not in names, "duplicate HFQ tensor name: " + name)
        names.add(name)
        qt, rank = unpack("<BB")
        shape = list(unpack("<%dI" % rank))
        group, size = unpack("<IQ")
        require(position + size <= file_size, "tensor exceeds artifact: " + name)
        entries.append(dict(name=name, qt=qt, shape=shape, group=group,
                            bytes=size, artifact_offset=position))
        position += size
    require(position == file_size, "HFQ payload end differs from artifact length")
    return entries


def select(entries):
    layers = {0: {}, 47: {}}
    for entry in entries:
        match = NAME.fullmatch(entry["name"])
        if not match:
            require(entry["qt"] not in (135, 136), "E8P tensor has unrecognized expert name")
            continue
        layer, expert = map(int, match.group(1, 2))
        kind = match.group(3)
        if layer in layers:
            layers[layer].setdefault(expert, {})[kind] = entry
    samples = []
    for layer, experts in layers.items():
        cold = []
        for expert, projections in sorted(experts.items()):
            if not any(t["qt"] in (135, 136) for t in projections.values()):
                continue
            require(0 <= expert < 512, "expert ID outside expected range")
            require(set(projections) == set(KINDS), "unpaired E8P projections")
            for kind, (qt, shape) in KINDS.items():
                tensor = projections[kind]
                numel = shape[0] * shape[1]
                expected_size = (numel // 8) * 2 + (numel // 128) * 2
                require((tensor["qt"], tensor["shape"], tensor["group"], tensor["bytes"])
                        == (qt, shape, 128, expected_size),
                        "invalid E8P qtype/shape/group/length: " + tensor["name"])
            cold.append(expert)
        require(len(cold) >= 2, "need two cold E8P experts in layer %d" % layer)
        for expert in (cold[0], cold[-1]):
            samples.append((layer, expert, experts[expert]))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--expect-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-fA-F]{64}", args.expect_sha256), "expected SHA256 must be 64 hex digits")
    artifact = args.artifact.resolve(strict=True)
    source = Path(__file__).resolve(strict=True)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    with artifact.open("rb") as stream:
        before = os.fstat(stream.fileno())
        artifact_sha = hash_range(stream, 0, before.st_size)
        require(artifact_sha == args.expect_sha256.lower(), "artifact SHA256 mismatch")
        selected = select(parse(stream, before.st_size))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        samples = []
        for layer, expert, projections in selected:
            tensors = []
            for kind in KINDS:
                entry = projections[kind]
                filename = "L%d.E%d.%s.bin" % (layer, expert, kind)
                destination = args.output_dir / filename
                stream.seek(entry["artifact_offset"])
                remaining = entry["bytes"]
                digest = hashlib.sha256()
                with destination.open("xb") as output:
                    while remaining:
                        block = stream.read(min(CHUNK, remaining))
                        require(block, "unexpected EOF during sample extraction")
                        output.write(block)
                        digest.update(block)
                        remaining -= len(block)
                sha = digest.hexdigest()
                require(hash_range(stream, entry["artifact_offset"], entry["bytes"]) == sha,
                        "artifact sample reread SHA256 mismatch")
                with destination.open("rb") as output:
                    require(os.fstat(output.fileno()).st_size == entry["bytes"]
                            and hash_range(output, 0, entry["bytes"]) == sha,
                            "written sample reread SHA256 mismatch")
                tensors.append(dict(kind=kind, qt=entry["qt"], shape=entry["shape"],
                                    group=entry["group"], file=filename, sha256=sha,
                                    artifact_offset=entry["artifact_offset"], bytes=entry["bytes"]))
            samples.append(dict(layer=layer, expert=expert, hot=False, tensors=tensors))
        after = os.fstat(stream.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "artifact changed during extraction")
    manifest = dict(artifact_path=str(artifact), artifact_sha256=artifact_sha,
                    provisional=False, samples=samples,
                    extractor_source_path=str(source), extractor_source_sha256=source_sha,
                    argv=[sys.executable, *sys.argv], rc=0, device="cpu")
    with (args.output_dir / "sample.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(args.output_dir / "sample.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, struct.error) as error:
        print("E8P sample extraction failed: %s" % error, file=sys.stderr)
        sys.exit(1)
