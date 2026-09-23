#!/usr/bin/env python3
"""Move redundant gfx1201 IU4 LOAD drains through an already-split barrier.

Matches instruction/dataflow structure, never PCs or physical register numbers.
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
    out = set()
    for match in REG_RE.finditer(text):
        if match.group(3) is not None:
            out.add(int(match.group(3)))
        else:
            out.update(range(int(match.group(1)), int(match.group(2)) + 1))
    return out

def operands(args: str) -> list[str]:
    return [part.strip() for part in args.split(",")]

def source_regs(insn: Insn) -> set[int]:
    parts = operands(insn.args)
    if not parts:
        return set()
    if insn.op.startswith(("global_load", "buffer_load", "flat_load", "scratch_load", "ds_load")):
        return regs(",".join(parts[1:]))
    if insn.op.startswith(("ds_store", "global_store", "buffer_store", "flat_store")):
        return regs(",".join(parts[1:]))
    if insn.op.startswith("v_"):
        return regs(",".join(parts[1:]))
    return regs(insn.args)

def defined_regs(insn: Insn) -> set[int]:
    if insn.op.startswith(("global_load", "buffer_load", "flat_load", "scratch_load", "ds_load", "v_")):
        parts = operands(insn.args)
        return regs(parts[0]) if parts else set()
    return set()

def load_dest(insn: Insn) -> set[int]:
    return regs(operands(insn.args)[0])

insns = []
function = "outside"
for line_number, raw in enumerate(lines):
    text = raw.strip()
    match = FUNC_RE.match(text)
    if match:
        function = match.group(1)
        continue
    if not text or text.startswith((";", ".")) or text.endswith(":"):
        continue
    fields = text.split(None, 1)
    if fields[0][0].isalpha():
        insns.append(Insn(line_number, fields[0], fields[1] if len(fields) == 2 else "", function))

def following(position: int, op: str, limit: int) -> int:
    for candidate in range(position + 1, min(len(insns), position + limit + 1)):
        if insns[candidate].function != insns[position].function:
            break
        if insns[candidate].op == op:
            return candidate
    raise SystemExit(f"no following {op} from line {insns[position].line + 1}")

def preceding(position: int, op: str, limit: int) -> int:
    for candidate in range(position - 1, max(-1, position - limit - 1), -1):
        if insns[candidate].function != insns[position].function:
            break
        if insns[candidate].op == op:
            return candidate
    raise SystemExit(f"no preceding {op} from line {insns[position].line + 1}")

deletions = set()
inserts_before = defaultdict(list)
audits = []
for position, insn in enumerate(insns):
    if insn.op != "s_wait_loadcnt" or int(insn.args, 0) != 0:
        continue

    signal = next(
        (
            candidate
            for candidate in range(position + 1, min(len(insns), position + 4))
            if insns[candidate].function == insn.function
            and insns[candidate].op == "s_barrier_signal"
        ),
        None,
    )
    if signal is None:
        continue
    barrier_wait = next(
        (
            candidate
            for candidate in range(signal + 1, min(len(insns), signal + 65))
            if insns[candidate].function == insn.function
            and insns[candidate].op == "s_barrier_wait"
        ),
        None,
    )
    if barrier_wait is None or barrier_wait - signal <= 8:
        continue
    global_inv = next(
        (
            candidate
            for candidate in range(barrier_wait + 1, min(len(insns), barrier_wait + 3))
            if insns[candidate].function == insn.function
            and insns[candidate].op == "global_inv"
        ),
        None,
    )
    if global_inv is None:
        continue
    branch = next(
        (
            candidate
            for candidate in range(global_inv + 1, min(len(insns), global_inv + 129))
            if insns[candidate].function == insn.function
            and insns[candidate].op == "s_cbranch_vccnz"
        ),
        None,
    )
    if branch is None:
        raise SystemExit(f"missing post-barrier branch in {insn.function}")

    # The next loop body's six-packet issue stream identifies the physical
    # destinations kept live across this split barrier without naming them.
    future_loads = [
        candidate
        for candidate in range(branch + 1, min(len(insns), branch + 601))
        if insns[candidate].function == insn.function
        and insns[candidate].op.startswith("global_load_")
    ]
    loads = None
    expected = ["global_load_b64"] * 4 + ["global_load_b32"] * 2
    for start in range(len(future_loads) - 5):
        window = future_loads[start : start + 6]
        if [insns[candidate].op for candidate in window] == expected:
            loads = window
            break
    if loads is None:
        raise SystemExit(f"no six-packet successor stream in {insn.function}")
    destinations = set().union(*(load_dest(insns[candidate]) for candidate in loads))

    for candidate in range(position + 1, branch + 1):
        if source_regs(insns[candidate]) & destinations:
            raise SystemExit(
                f"LOAD destination used before relocated wait in {insn.function} "
                f"at line {insns[candidate].line + 1}"
            )
        if defined_regs(insns[candidate]) & destinations:
            raise SystemExit(
                f"LOAD destination overwritten before relocated wait in {insn.function} "
                f"at line {insns[candidate].line + 1}"
            )

    # The six packet stream is older than this iteration's GLOBAL_INV_ACCESS.
    # No destination is touched before the branch, so loadcnt(1) is the weakest
    # safe wait: only the younger invalidate may remain outstanding.
    deletions.add(insn.line)
    inserts_before[insns[branch].line].append("\ts_wait_loadcnt 0x1\n")
    audits.append(
        f"RULE_SPLIT {insn.function} line {insn.line + 1}: "
        "move pre-signal loadcnt(0) through signal/wait+fold to pre-branch loadcnt(1); "
        "six packets, one younger GLOBAL_INV_ACCESS"
    )

if len(audits) != 4:
    raise SystemExit(f"RULE_SPLIT expected 4 loop-path matches, found {len(audits)}")

out = []
for line_number, raw in enumerate(lines):
    out.extend(inserts_before.get(line_number, ()))
    if line_number not in deletions:
        out.append(raw)
output_path.write_text("".join(out))
print("\n".join(audits))
