#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Gate G2 — is the Ornith 1.5 MTP module's norm set actually trained?

ORNITH 1.0's DSpark drafter shipped trained matmuls beside completely
untrained RMSNorms (every weight exactly 1.0, std 0.0). That produced
tau=0.00 and cost days of engine debugging before the checkpoint itself was
identified as the defect. hipfire's math was correct to cosine 0.999873.

A trained reference measured norm mean 1.52 / std 0.14 — but that figure comes
from a classic weight-near-1 RMSNorm. THIS checkpoint family uses a zero-centred
(1+w) parameterisation, so its trained norms sit near 0.0, not near 1.0 (verified
against the model body's own certainly-trained norms: means 0.0311, -0.1052).
Do not judge trained-ness by the mean here; judge it by per-element variance.

Exit 1 here means: drop MTP from scope and report upstream. Do not debug.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Deliberately NO `safetensors` / `ml_dtypes` dependency. Neither is present in
# this host's default python3, so importing them made the gate unrunnable by the
# very command its own docs prescribe — a gate you cannot re-run decays into a
# claim. The safetensors container is trivial to parse (u64 header length, JSON
# header, raw tensor bytes) and bf16 -> f32 is an exact 16-bit left shift, so
# both dependencies are avoidable at no cost in correctness.

DEFAULT_SRC = "/home/nick/hf/Ornith-1.5-35B-A3B"
FROZEN_STD_EPS = 1e-6


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(
            "usage: python3 -m tools.models.ornith.mtp_norm_check [model_dir]\n"
            f"  default model_dir: {DEFAULT_SRC}"
        )
        return 0

    src = Path(args[0] if args else DEFAULT_SRC)
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    norm_keys = sorted(k for k in wm if k.startswith("mtp.") and "norm" in k
                       and k.endswith(".weight"))
    matmul_keys = sorted(k for k in wm if k.startswith("mtp.")
                         and k.endswith(("fc.weight", "q_proj.weight", "o_proj.weight")))

    if not norm_keys:
        print("FAIL: no mtp.* norm tensors found; the module layout is not what we assume")
        return 1

    _shards = {}

    def _header(shard):
        """Parse a safetensors container: u64 header length, JSON header, payload."""
        if shard not in _shards:
            fh = open(src / shard, "rb")
            hlen = struct.unpack("<Q", fh.read(8))[0]
            _shards[shard] = (json.loads(fh.read(hlen)), 8 + hlen, fh)
        return _shards[shard]

    def load(key):
        meta, payload_off, fh = _header(wm[key])
        entry = meta[key]
        start, end = entry["data_offsets"]
        fh.seek(payload_off + start)
        raw = fh.read(end - start)
        dtype = entry["dtype"]
        if dtype == "BF16":
            # bf16 is the top 16 bits of an f32 — widening is exact, never lossy.
            return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        if dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32)
        if dtype == "F16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        # Fail closed: an unhandled dtype must not be silently skipped, because a
        # skipped norm is an unexamined norm and this is a gate.
        print(f"FAIL: unhandled dtype {dtype!r} for {key}")
        sys.exit(1)

    def summarize(keys, label):
        rows = []
        for k in keys:
            t = load(k)
            rows.append((k, float(t.mean()), float(t.std())))
        print(f"\n{label} ({len(rows)} tensors):")
        for k, m, s in rows[:6]:
            print(f"  {k:<62} mean={m:8.4f} std={s:8.4f}")
        if len(rows) > 6:
            print(f"  ... {len(rows) - 6} more")
        return rows

    norms = summarize(norm_keys, "MTP learnable RMSNorm weights")
    mms = summarize(matmul_keys, "MTP matmul weights (trained-ness control)")

    # Frozen-norm detection, convention-INDEPENDENT.
    #
    # Do NOT test "mean ~ 1.0". This checkpoint family uses a zero-centred (1+w)
    # RMSNorm parameterisation — verified 2026-08-20 against the model body's own
    # certainly-trained norms, whose means sit near 0.03 / -0.11, not near 1.0.
    # Under (1+w) the identity element is w=0.0, so an untrained export freezes at
    # mean 0.0, and a mean-vs-1.0 test can never fire: the gate would return a
    # vacuous PASS on precisely the defect it exists to catch.
    #
    # Near-zero per-element variance is the real signature. A trained norm never
    # has vanishing spread, whatever constant it is centred on.
    #
    # EPSILON rather than `== 0.0`: bit-exact equality only catches a perfectly
    # uniform freeze. A norm frozen at a constant but carrying a few denormal or
    # rounding-noise elements would score std ~1e-9 and slip through the same way
    # the old mean-anchored test did. The margin is enormous — the smallest real
    # std measured on this checkpoint is 2e-4, i.e. 200x above this threshold — so
    # the epsilon costs no false positives while removing a dependence on exact
    # floating-point summation that would not survive a differently-shaped export.
    degenerate = [(k, m, s) for k, m, s in norms if s < FROZEN_STD_EPS]
    frac = len(degenerate) / len(norms)

    print(f"\nFrozen (std < {FROZEN_STD_EPS:g}) norms: {len(degenerate)}/{len(norms)} "
          f"({frac:.1%})")
    for k, m, s in degenerate:
        print(f"  FROZEN {k} mean={m:.6f}")

    # Control: if the matmuls are ALSO degenerate, the export is broken wholesale
    # rather than norm-specific, which is a different report.
    mm_alive = [s for _, _, s in mms if s > 1e-4]
    print(f"Matmuls with non-trivial std: {len(mm_alive)}/{len(mms)}")

    # ANY frozen norm fails, not a majority. There are only ~7 norm tensors in a
    # one-layer MTP module; a single dead sub-component (q_norm alone, say) is
    # enough to break the drafter. A `frac > 0.5` threshold would let three of
    # seven be completely frozen and still report PASS.
    if degenerate:
        print("\nG2 FAIL — MTP norms are frozen (untrained export).")
        if mms and len(mm_alive) == len(mms):
            print("Matmuls ARE trained, so this is the norm-specific defect, not an empty export.")
        print("ACTION: drop the MTP sidecar from scope. Report upstream. Do NOT debug engine code.")
        return 1

    # The matmul arm must gate too, not merely decorate the failure message: an
    # export with live norms but dead matmuls is equally unusable, and testing
    # only norms would pass it.
    if mms and not mm_alive:
        print("\nG2 FAIL — every MTP matmul is degenerate; the export is empty.")
        return 1

    print("\nG2 PASS — MTP norms carry trained variance. Task 8 may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
