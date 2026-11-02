#!/usr/bin/env python3
"""Resolve a PCI address to the HIP device index for this boot.

HIP enumerates GPUs in KFD-node order, which is NOT PCI bus order and is not
stable across boots or hotplug. On hiptrx the order is::

    0=03:00.0  1=c3:00.0  2=e3:00.0  3=4b:00.0  4=13:00.0

so the gate's historical `device: "3"` resolved to `4b:00.0` -- the
Thunderbolt-attached R9700 on an x4 link -- and every gfx1201 lane silently
benchmarked the eGPU rather than a mainboard card. When that card
re-enumerated (7b -> 4b, after a tunnel cycle) the whole map shifted again,
so the same literal index pointed at a different GPU.

A lane therefore pins a PCI address and resolves it here, at lane start. A
pinned card that is absent is a hard error: benchmarking whichever GPU happens
to occupy an index is worse than not running.

Node properties come from `/sys/class/kfd/kfd/topology/nodes/*/properties`:
`location_id` packs the PCI bus and device as `(bus << 8) | device`, and
`simd_count == 0` marks the CPU node, which is not a HIP device.

Usage:
    resolve_device.py --pci 0000:c3:00.0            # prints HW_GATE_DEVICE=1
    resolve_device.py --pci c3:00.0 --list          # also lists the whole map
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

TOPOLOGY_ROOT = "/sys/class/kfd/kfd/topology/nodes"

_PCI_RE = re.compile(r"^(?:([0-9a-f]{4}):)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f])$")


def parse_pci(addr: str) -> int:
    """`[domain:]bus:device.function` -> KFD `location_id` ((bus << 8) | device)."""
    m = _PCI_RE.match(addr.strip().lower())
    if not m:
        raise ValueError(f"not a PCI address: {addr!r}")
    return (int(m.group(2), 16) << 8) | int(m.group(3), 16)


def gfx_name(target_version: int) -> str:
    """KFD `gfx_target_version` -> arch name.

    `major * 10000 + minor * 100 + step`, printed with hex digits for minor and
    step so the CDNA `gfx90a`-style names come out right. Verified against both
    gate hosts: 120001 -> gfx1201, 110000 -> gfx1100, 110501 -> gfx1151,
    100300 -> gfx1030.
    """
    major, minor, step = target_version // 10000, (target_version // 100) % 100, target_version % 100
    return f"gfx{major}{minor:x}{step:x}"


def gpu_nodes(root: str = TOPOLOGY_ROOT) -> list[tuple[int, int, str]]:
    """[(hip_index, location_id, gfx)] in KFD-node order, CPU nodes skipped."""
    out: list[tuple[int, int, str]] = []
    paths = glob.glob(os.path.join(root, "*", "properties"))
    if not paths:
        raise FileNotFoundError(f"no KFD topology under {root} (is amdgpu loaded?)")

    def node_number(path: str) -> int:
        try:
            return int(os.path.basename(os.path.dirname(path)))
        except ValueError:
            return 1 << 30

    for path in sorted(paths, key=node_number):
        fields: dict[str, str] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(" ")
                if value:
                    fields[key.strip()] = value.strip()
        if int(fields.get("simd_count", "0") or 0) == 0:
            continue
        out.append((
            len(out),
            int(fields.get("location_id", "0") or 0),
            gfx_name(int(fields.get("gfx_target_version", "0") or 0)),
        ))
    return out


def fmt_location(location_id: int) -> str:
    return f"{location_id >> 8:02x}:{location_id & 0xFF:02x}.0"


def resolve(addr: str, root: str = TOPOLOGY_ROOT, expect_gfx: str | None = None) -> int:
    """PCI address -> HIP index, optionally asserting the card's arch.

    `expect_gfx` makes the pin self-checking: a wrong address and a swapped or
    re-slotted card both fail loudly instead of benchmarking the wrong GPU.
    It caught a real error while this was being written -- the hipx lane was
    first pinned to 0000:03:00.0, which is a hiptrx address; hipx's gfx1100 is
    at 0000:66:00.0.
    """
    want = parse_pci(addr)
    nodes = gpu_nodes(root)
    have = ", ".join(f"{i}={fmt_location(loc)}({gfx})" for i, loc, gfx in nodes)
    for index, location_id, gfx in nodes:
        if location_id != want:
            continue
        if expect_gfx and gfx != expect_gfx:
            raise LookupError(
                f"pinned card {addr} is {gfx}, not the expected {expect_gfx} "
                f"(cards: {have}) -- the card at that address changed"
            )
        return index
    raise LookupError(f"pinned card {addr} is not among the {len(nodes)} GPU node(s): {have}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pci", required=True, help="PCI address to pin, e.g. 0000:c3:00.0")
    ap.add_argument("--topology-root", default=TOPOLOGY_ROOT, help="KFD topology nodes dir")
    ap.add_argument("--list", action="store_true", help="also print the full index map to stderr")
    ap.add_argument("--expect-gfx", default=None, help="assert the pinned card's arch, e.g. gfx1201")
    args = ap.parse_args(argv)
    try:
        index = resolve(args.pci, args.topology_root, args.expect_gfx)
        if args.list:
            nodes = gpu_nodes(args.topology_root)
            sys.stderr.write(
                "HIP index map: " + ", ".join(f"{i}={fmt_location(loc)}({gfx})" for i, loc, gfx in nodes) + "\n"
            )
    except (ValueError, LookupError, FileNotFoundError) as exc:
        sys.stderr.write(f"resolve_device: {exc}\n")
        return 2
    print(f"HW_GATE_DEVICE={index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
