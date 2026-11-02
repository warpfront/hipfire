# MQ-I4 A4 K3 gfx1100: G1 byte oracle (standalone qualification)

Byte-exact acceptance of the gfx1100 K3 producer `mq_i4_a4_r128_k640_gfx1100` (emitter output
`kernels/mq_i4_a4_pm_gfx1100.hxaco`). Standalone qualification only: no registry, dispatch or Cargo.toml wiring.
Expected device: `GPU-43390a851e296ee5`, BDF `0000:67:00.0`, `gfx1100`.

**Link caveat.** The card is an x16 endpoint behind a PCIe 4.0 x4 root-port link. Every timing and bandwidth figure
carries that limitation. Timing here is per-iteration host wall clock around `launch_kernel_blob` + `device_synchronize`
of the K3 kernel alone (launch + sync included, synchronized per loop). It is not an end-to-end claim.

## Files

| file | role |
|---|---|
| `crates/hipfire-runtime/examples/mq_i4_a4_gfx1100_g1_oracle.rs` | raw HIP runner (same 48-byte kernarg ABI, load and identity check as `mq_i4_a4_byte_oracle`, symbol `mq_i4_a4_r128_k640_gfx1100`) |
| `crates/peacemaker-isa/tools/mq_i4_a4_gfx1100_oracle.py` | CPU driver: case generation, expected bytes, comparison, receipt |
| `crates/peacemaker-isa/tools/mq_i4_a4_oracle.py` | certified base harness, imported (not copied) |
| `/home/kaden/qcal/release-0.4.2/subq4/calib2/pilot_solve.py` | CPU oracle `a4_emulate`; SHA256 pinned to `53965f2328e32d86c7b87460a02573a065424e220383891da210441dc9f6c0df` (any other file is refused). The runtime `mq_i4_reference` has no A4 producer. |

The runner is an auto-discovered example. K3 takes BF16 activations; it takes no weight scales. The code x f16-scale
inputs below are only structured BF16 patterns.

## Case set

* All base families unchanged: `rand10k`, `rand10k_sparse`, `extremes`, `zeros`, `shapes`, `chunks`, `signs`, `solo_*`,
  plus the 64-row batch-independence check (solo bytes equal the same row inside the 10k batch).
* `codescale`: BF16-rounded rows = signed code (-8..7) x f16 positive scale in {`0x0000`, `0x0001` (subnormal), `0x03ff`,
  `0x0400`, `0x3c00`, `0x7bff` (max)}. Cyclic, constant, 8-blocked and random code patterns per scale, a mixed-scale case
  (a different scale per K128 group) and one sparse-slot case over everything. Scale 0 with a negative code gives -0.0.
* Negative launches (separate dirs, deliberately mutated host inputs; expected is always the ORIGINAL case's CPU bytes,
  never derived from the GPU). Each must produce mismatch bytes > 0:
  * `neg_signs/`: `neg_signs_negate_s1`, `neg_signs_swap_s1_s2` (on `rand10k`)
  * `neg_sorted/`: `neg_sorted_rotate_slots` (`rand10k`), `neg_sorted_swap_pair` (`rand10k_sparse`),
    `neg_sorted_drop_valid` (`chunk4096_edges_valid`)
* Real inputs (optional; absent => receipt `real.status = "pending"`, never "passed"):
  * `--real-manifest PATH`: JSON `{artifact_sha256, layers, source_mode, cases:[{name, layer, expert, rows, h_path}]}`,
    `h_path` a raw little-endian BF16 `[rows, 640]` file (relative paths resolve against the manifest directory). Slots are
    the identity `[0..rows)`, `L = rows`, fixed sign tables (the base harness's `gen_fwht_signs(43)` / `(1043)`). Cases run as
    `real_<name>`. Label: "real-weight-derived inputs (ctl3-hot gate/up over deterministic A4)". These are not XTS-route
    and not real-distribution measurements. Manifest SHA256, artifact SHA256, layer, expert and each `h` file hash are recorded.
  * `--real-h PATH --real-sorted PATH --real-l N`: raw LE BF16 `[P, 640]`, raw LE i32 `[P]` (-1 = pad), slot count `N`
    (case `real_down_input`; real sample layout is the caller's to resolve).

## Work directory layout (`--work W`)

```
W/base/ W/neg_signs/ W/neg_sorted/   runner dirs: manifest.json, N.{h,sorted,signs1,signs2}.bin; runner writes N.xq.out.bin
W/expected/N.xq.expected.bin         CPU expected bytes of base cases (compare host only; recomputed if absent)
W/prepare.json                       case metadata, input/expected hashes, mutation records, real inputs, tool hashes
W/run_runner.sh                      relocatable (W derived from the script dir): sh W/run_runner.sh RUNNER CODEOBJECT
W/receipts/{base,neg_signs,neg_sorted}.{cmd,rc,stderr,runner.json}   exact command, rc, runner receipt per dir
W/receipts/{runner,codeobject}.sha256                                 hashes measured where the job ran
```

`run_runner.sh` has the identity expectation (`--expect-uuid/--expect-bdf`) and the timing flags baked in and sets no
device-visibility or CU-mask variable. The driver refuses to run if any environment variable containing `CU_MASK` is set.

## Invocation

Build (runs nowhere near a GPU; no CU-mask variables; the oracle runner requires the existing `lab` feature, while the `mq_i4_a4_gfx1100_emit` native emitter keeps `--features toolchain`):

```
env ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 \
  CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.3/target-consumer24-k3 \
  nice -n19 cargo build -j8 -p hipfire-runtime --release --features lab --example mq_i4_a4_gfx1100_g1_oracle
```

Split flow (CPU expected generation local, GPU run remote on the fabric):

```
# 1. local CPU
python3 crates/peacemaker-isa/tools/mq_i4_a4_gfx1100_oracle.py --prepare-only \
  --work W --receipt /home/kaden/qcal/release-0.4.3/consumer24/rcpt/b3.g1.prepare.json \
  [--iterations 1 --warmup 0] [--real-manifest M.json] [--emitter-source PATH ...]
# 2. transfer W (expected/ is not needed remotely), then ONE fabric job on the card (run ON hipx via ssh):
/home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card xtx --mem-gb 2 --priority P1 \
  --owner <owner> --max-secs 1800 -- sh W/run_runner.sh /abs/mq_i4_a4_gfx1100_g1_oracle /abs/mq_i4_a4_pm_gfx1100.hxaco
# 3. transfer W back, local CPU
python3 crates/peacemaker-isa/tools/mq_i4_a4_gfx1100_oracle.py --compare-only \
  --work W --receipt /home/kaden/qcal/release-0.4.3/consumer24/rcpt/b3.g1.json \
  [--runner BIN --codeobject CO]   # optional local hash cross-check against the remote-measured hashes
```

Single-shot (prepare + run + compare in one process, when the CPU libs and the card share a host):

```
python3 .../mq_i4_a4_gfx1100_oracle.py --work W --receipt .../b3.g1.json \
  --runner /abs/mq_i4_a4_gfx1100_g1_oracle --codeobject /abs/mq_i4_a4_pm_gfx1100.hxaco \
  --runner-prefix "/home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card xtx --mem-gb 2 --priority P1 --owner <owner> --max-secs 1800 --"
```

Other flags: `--selftest` (CPU only), `--rows N` (>= 10000), `--seed`, `--pilot-solve PATH` (must hash to the pin),
`--base-harness PATH`, `--expect-base-sha256`, `--expect-uuid/--expect-bdf` (default the pinned card),
`--recompute-expected`. Run the runner by hand: `mq_i4_a4_gfx1100_g1_oracle --codeobject CO --cases DIR [--expect-uuid U]
[--expect-bdf B] [--iterations N [--warmup M]]`.

Timing: `--iterations` (default 1; 0 disables) launches are timed after the correctness launch and its read-back, on the
`base/` dir only. For real timing use a separate `--work` and a `--class timing` fabric job with larger `--iterations/--warmup`.

## Receipt and verdict

`b3.*.json` (written on every path, including runner failure) records: argv, env visibility, tool/library versions, SHA256
of this driver, the runner source, the runner binary, the codeobject, any `--emitter-source`, the pilot and the base
harness; per case input and expected/output hashes, mismatch counts and first mismatching records; families, batch
independence; per negative the mutated inputs, mismatch bytes and `pass`; per runner dir the exact command, rc, stderr tail
and runner receipt (device uuid/bdf/arch); the timing block; the real block; and the link caveat.

Verdict: `PASS` (runner jobs ok, device identity pinned, all synthetic cases and independence pass, all negatives mismatch,
real inputs passed), `PASS_SYNTHETIC_REAL_PENDING` (same, no real input supplied; `pass` stays false), or `FAIL`. Exit
status is 0 for the first two only.
