#!/usr/bin/env python3
"""Apply counter-ledger rules to the emitted gfx1201 IU4 assembly.

The rules are derived from instruction/dataflow patterns, not PCs or physical
register numbers.  PCs are deliberately left to the post-assembly audit.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys

if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s")

input_path, output_path = map(Path, sys.argv[1:])
lines = input_path.read_text().splitlines(keepends=True)
REG_RE = re.compile(r"\bv(?:\[(\d+):(\d+)\]|(\d+))(?!\w)")
FUNC_RE = re.compile(r"^(gemm_mq4g256v2_residual_mmq_iu4(?:_full_(?:add|set))?):")


@dataclass(frozen=True)
class Insn:
    line: int
    op: str
    args: str
    function: str


def regs(text: str) -> set[int]:
    found: set[int] = set()
    for match in REG_RE.finditer(text):
        if match.group(3) is not None:
            found.add(int(match.group(3)))
        else:
            lo, hi = int(match.group(1)), int(match.group(2))
            found.update(range(lo, hi + 1))
    return found


def operands(args: str) -> list[str]:
    return [part.strip() for part in args.split(",")]


def source_regs(insn: Insn) -> set[int]:
    parts = operands(insn.args)
    if not parts:
        return set()
    if insn.op.startswith("global_load"):
        return regs(",".join(parts[1:]))
    if insn.op.startswith(("ds_store", "global_store")):
        return regs(",".join(parts[1:]))
    if insn.op.startswith(("v_", "ds_load")):
        return regs(",".join(parts[1:]))
    return regs(insn.args)


def load_dest(insn: Insn) -> set[int]:
    return regs(operands(insn.args)[0])


def is_load_counter_event(insn: Insn) -> bool:
    """gfx12 LOAD_CNT events from the M04/LLVM event map."""
    return insn.op == "global_inv" or insn.op.startswith(
        ("global_load_", "buffer_load_", "flat_load_", "scratch_load_", "global_atomic_")
    )


def younger_load_events(producer: int, wait_position: int) -> int:
    """Count LOAD_CNT events issued after producer and before the wait."""
    return sum(
        is_load_counter_event(insns[candidate])
        for candidate in range(producer + 1, wait_position)
    )

def wait_count(insn: Insn) -> int:
    return int(insn.args, 0)


insns: list[Insn] = []
function = "outside"
for line_number, raw in enumerate(lines):
    text = raw.strip()
    function_match = FUNC_RE.match(text)
    if function_match:
        function = function_match.group(1)
        continue
    if not text or text.startswith((";", ".")) or text.endswith(":"):
        continue
    fields = text.split(None, 1)
    if not fields[0][0].isalpha():
        continue
    insns.append(Insn(line_number, fields[0], fields[1] if len(fields) == 2 else "", function))

replacements: dict[int, str] = {}
deletions: set[int] = set()
inserts_before: dict[int, list[str]] = defaultdict(list)
inserts_after: dict[int, list[str]] = defaultdict(list)
audits: list[str] = []


def preceding(position: int, op: str) -> int:
    for candidate in range(position - 1, -1, -1):
        if insns[candidate].function != insns[position].function:
            break
        if insns[candidate].op == op:
            return candidate
    raise ValueError(f"no preceding {op} at source line {insns[position].line + 1}")


def following(position: int, op: str, limit: int = 64) -> int:
    for candidate in range(position + 1, min(len(insns), position + limit + 1)):
        if insns[candidate].function != insns[position].function:
            break
        if insns[candidate].op == op:
            return candidate
    raise ValueError(f"no following {op} at source line {insns[position].line + 1}")


def first_direct_use(load_position: int, stop_position: int) -> int | None:
    destination = load_dest(insns[load_position])
    for candidate in range(load_position + 1, stop_position):
        if source_regs(insns[candidate]) & destination:
            return candidate
    return None


# Rule 1: a full LOAD drain immediately before a workgroup signal is movable
# when all still-pending load destinations are first consumed after the paired
# wait. Place a wait at each consuming instruction. The weakest safe count is
# the number of younger LOAD_CNT events already issued: later load packets plus
# GLOBAL_INV_ACCESS events (which share LOAD_CNT on gfx12).
primary_clusters = 0
for position, insn in enumerate(insns):
    if insn.op != "s_wait_loadcnt" or wait_count(insn) != 0:
        continue
    if position + 1 >= len(insns) or insns[position + 1].op != "s_barrier_signal":
        continue
    barrier_wait = following(position, "s_barrier_wait", 12)
    global_inv = following(barrier_wait, "global_inv", 3)
    previous_barrier = preceding(position, "s_barrier_wait")
    loads = [
        candidate
        for candidate in range(previous_barrier + 1, position)
        if insns[candidate].op == "global_load_b64"
    ]
    if len(loads) != 4:
        continue
    stores = [
        candidate
        for candidate in range(global_inv + 1, min(len(insns), global_inv + 8))
        if insns[candidate].op == "ds_store_2addr_stride64_b64"
    ]
    if len(stores) < 2:
        continue
    stores = stores[:2]

    # Validate that already-consumed packets had legal partial waits and that
    # the remaining packets' first direct uses are the post-barrier stores.
    first_uses = [first_direct_use(load, stores[-1] + 1) for load in loads]
    if first_uses[0] is None or first_uses[1] is None:
        raise SystemExit(f"missing pre-barrier packet use in {insn.function}")
    for packet_index in (0, 1):
        use = first_uses[packet_index]
        prior_waits = [
            candidate
            for candidate in range(loads[packet_index] + 1, use)
            if insns[candidate].op == "s_wait_loadcnt"
        ]
        if not prior_waits:
            raise SystemExit(f"packet {packet_index + 1} lacks a LOAD wait in {insn.function}")
        actual = wait_count(insns[prior_waits[-1]])
        required = younger_load_events(loads[packet_index], use)
        if actual > required:
            raise SystemExit(
                f"packet {packet_index + 1} under-waited in {insn.function}: "
                f"loadcnt({actual}) > {required}"
            )

    store_packets: list[int] = []
    for store in stores:
        matching = [
            packet
            for packet, load in enumerate(loads, start=1)
            if source_regs(insns[store]) & load_dest(insns[load])
        ]
        if not matching:
            raise SystemExit(f"post-barrier store has no direct load producer in {insn.function}")
        store_packets.append(max(matching))
    if store_packets != sorted(store_packets) or store_packets[0] <= 2:
        raise SystemExit(f"unexpected post-barrier packet order {store_packets} in {insn.function}")

    counts = [
        younger_load_events(loads[packet - 1], store)
        for store, packet in zip(stores, store_packets)
    ]
    deletions.add(insn.line)
    for store, count in zip(stores, counts):
        inserts_before[insns[store].line].append(f"\ts_wait_loadcnt {count:#x}\n")
    audits.append(
        f"RULE1 {insn.function} line {insn.line + 1}: move loadcnt(0) across barrier; "
        f"packet uses {store_packets} -> counts {counts}"
    )
    primary_clusters += 1

if primary_clusters != 4:
    raise SystemExit(f"RULE1 expected 4 kernel-body matches, found {primary_clusters}")


# Rule 2: when a mixed zero wait follows the last DS publication store and is
# immediately followed by a workgroup signal, LOAD is already covered by Rule
# 1.  Preserve only the DS prerequisite.
publication_clusters = 0
for position, insn in enumerate(insns):
    if insn.op != "s_wait_loadcnt_dscnt" or wait_count(insn) != 0:
        continue
    if not (position and position + 1 < len(insns)):
        continue
    if insns[position - 1].op != "ds_store_2addr_stride64_b64":
        continue
    if insns[position + 1].op != "s_barrier_signal":
        continue
    replacements[insn.line] = "\ts_wait_dscnt 0x0\n"
    audits.append(
        f"RULE2 {insn.function} line {insn.line + 1}: mixed(0) -> dscnt(0) after DS publication"
    )
    publication_clusters += 1

if publication_clusters != 4:
    raise SystemExit(f"RULE2 expected 4 kernel-body matches, found {publication_clusters}")


# Rule 3: split a mixed wait before a workgroup barrier when its LOAD packets
# are not consumed before the barrier. Keep dscnt(0) before the signal and put
# the weakest LOAD wait after global_inv. Its count includes every younger
# LOAD_CNT event already issued, including GLOBAL_INV_ACCESS. Placement before
# any post-barrier branch is conservative on every CFG successor.
next_slab_clusters = 0
for position, insn in enumerate(insns):
    if insn.op != "s_wait_loadcnt_dscnt" or wait_count(insn) != 0:
        continue
    if not (position and position + 1 < len(insns)):
        continue
    if insns[position - 1].op != "ds_load_2addr_b32" or insns[position + 1].op != "s_barrier_signal":
        continue
    previous_barrier = preceding(position, "s_barrier_wait")
    loads = [
        candidate
        for candidate in range(previous_barrier + 1, position)
        if insns[candidate].op.startswith("global_load_")
    ]
    wmma_count = sum(
        insns[candidate].op.startswith("v_wmma_")
        for candidate in range(previous_barrier + 1, position)
    )
    if len(loads) != 6 or wmma_count < 8:
        continue
    barrier_wait = following(position, "s_barrier_wait", 3)
    global_inv = following(barrier_wait, "global_inv", 2)

    first_use = None
    first_packets: list[int] = []
    for candidate in range(global_inv + 1, min(len(insns), global_inv + 80)):
        used = [
            packet
            for packet, load in enumerate(loads, start=1)
            if source_regs(insns[candidate]) & load_dest(insns[load])
        ]
        if used:
            first_use = candidate
            first_packets = used
            break
    if first_use is None:
        raise SystemExit(f"no post-barrier LOAD consumer in {insn.function}")
    youngest_producer = loads[max(first_packets) - 1]
    count = younger_load_events(youngest_producer, global_inv + 1)
    replacements[insn.line] = "\ts_wait_dscnt 0x0\n"
    inserts_after[insns[global_inv].line].append(f"\ts_wait_loadcnt {count:#x}\n")
    audits.append(
        f"RULE3 {insn.function} line {insn.line + 1}: split mixed(0); "
        f"first post-barrier packet(s) {first_packets}/6 -> loadcnt({count})"
    )
    next_slab_clusters += 1

if next_slab_clusters != 4:
    raise SystemExit(f"RULE3 expected 4 kernel-body matches, found {next_slab_clusters}")


out: list[str] = []
for line_number, raw in enumerate(lines):
    out.extend(inserts_before.get(line_number, ()))
    if line_number not in deletions:
        out.append(replacements.get(line_number, raw))
    out.extend(inserts_after.get(line_number, ()))
output_path.write_text("".join(out))
print("\n".join(audits))
