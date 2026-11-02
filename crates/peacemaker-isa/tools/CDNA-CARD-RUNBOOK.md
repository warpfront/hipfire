# CDNA card runbook: `cdna_card_tierc` on MI300X (gfx942)

Hardware proof that the three PeaceMaker gfx942 kernels (HIP twins) behave on a
real MI300X the way the CPU emulator says they do. The committed
[`docs/pm-cdna/card/`](../../../../docs/pm-cdna/card/) receipts cover
`60bfa12b8af3d21ab311d8b7b2052b7fcd6409fe`: quick 18/18 and full 47/47 pass.
These are correctness receipts, not throughput certification or runtime admission.

Kernels (twins in `kernels/src/pm_cdna/<name>.gfx942.hip`):

| Kernel | Twin |
|---|---|
| `fp8_act_producer` | `fp8_act_producer.gfx942.hip` |
| `fp8_moe_gate_up` | `fp8_moe_gate_up.gfx942.hip` |
| `qsa_fp8_kv` (encode and decode) | `qsa_fp8_kv.gfx942.hip` |

**The box builds from source. No binary is shipped.** The package is a
`git archive` of one commit; `card_run.sh` builds the example and the code
objects on the card.

## 0. Prerequisite: the harness must be committed

`cdna_card_package.sh` archives a commit, not the working tree. The example
`crates/peacemaker-isa/examples/cdna_card_tierc.rs`, the three twins, both
Cargo manifests/lockfile and these two tool files must be committed at the SHA
being packaged. The script refuses (non-zero exit, names the missing files) if
any are absent from that commit. Commit first, then package. Uncommitted edits
are silently not part of the archive (the script warns if the worktree differs).

## 1. Package (dev machine, no GPU)

```sh
cd /home/kaden/qcal/release-0.4.2/pm-cdna3
crates/peacemaker-isa/tools/cdna_card_package.sh /home/kaden/qcal/release-0.4.2/pm-cdna3-ref/card-package            # SHA defaults to HEAD
crates/peacemaker-isa/tools/cdna_card_package.sh /home/kaden/qcal/release-0.4.2/pm-cdna3-ref/card-package <sha>      # or an explicit commit
```

Emits in OUTDIR (`<sha>` is the full 40-hex commit):

- `pmcdna-<sha>.tar`: `git archive --format=tar --prefix=pmcdna-<sha>/ <sha>`
- `card_run.sh`: executable, `<sha>` baked in
- `SHA256SUMS`: checksums of the tar and `card_run.sh`

## 2. Transfer

```sh
scp /home/kaden/qcal/release-0.4.2/pm-cdna3-ref/card-package/{pmcdna-<sha>.tar,card_run.sh,SHA256SUMS} <user>@<card-host>:<dir>/
```

Keep the three files together in one directory on the card.

## 3. Run on the card

```sh
cd <dir>
bash card_run.sh
```

`card_run.sh` does, in order, stopping at the first failure:

1. `sha256sum -c SHA256SUMS`
2. extract the tar into `/scratch/g942/pmcdna/<sha>/` (`--strip-components=1`)
3. record `cargo`/`rustc`/`hipcc`/`/opt/rocm/lib/llvm/bin/clang` versions to `/scratch/g942/pmcdna/<sha>/toolchain.txt`
4. build with
   `RUSTUP_HOME=/scratch/g942/rustup CARGO_HOME=/scratch/g942/cargo PATH=/scratch/g942/cargo/bin:/opt/rocm/bin:$PATH CARGO_TARGET_DIR=/scratch/g942/target-pmcdna ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 nice -n 10 cargo build --release -p peacemaker-isa --features lab --example cdna_card_tierc -j 8`
5. compile each twin with
   `ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 nice -n 10 /opt/rocm/bin/hipcc --genco --offload-arch=gfx942 -O3 <twin>.gfx942.hip -o /scratch/g942/pmcdna/<sha>/hsaco/<name>.hsaco`
6. **quick** gate, one `g942run.sh` call, own receipt:
   ```sh
   G942_OWNER=PmCdnaLead /scratch/g942/g942run.sh pmcdna-tierc \
     timeout --signal=KILL 600 /scratch/g942/target-pmcdna/release/examples/cdna_card_tierc \
     --quick --hsaco-dir /scratch/g942/pmcdna/<sha>/hsaco \
     --receipt /scratch/g942/pmcdna/<sha>/cdna_card_tierc.quick.json
   ```
7. **full** gate, a separate `g942run.sh` call, own receipt:
   ```sh
   G942_OWNER=PmCdnaLead /scratch/g942/g942run.sh pmcdna-tierc \
     timeout --signal=KILL 600 /scratch/g942/target-pmcdna/release/examples/cdna_card_tierc \
     --hsaco-dir /scratch/g942/pmcdna/<sha>/hsaco \
     --receipt /scratch/g942/pmcdna/<sha>/cdna_card_tierc.json
   ```

Only steps 6 and 7 touch the GPU, and only through `g942run.sh`. Nothing is
written outside `/scratch/g942/pmcdna/<sha>/` except the shared cargo target
dir and the prescribed rust/cargo homes. The code objects are named
`<name>.hsaco`, which is what the example loads from `--hsaco-dir`.

`--dry-run` is GPU-free: no HIP library, no HIP call; emulator and CPU reference
only. Its card columns are a second emulator run, so a dry-run receipt is not
hardware evidence. When the HIP twin is unavailable, cardHIP is a synthetic
second PM output (`cardHIP_source` says so), so (a) then validates the harness
plumbing, not the twin; (d) stays unavailable with a reason. Dry-run is not part
of `card_run.sh`.

Expected environment: ROCm HIP 7.16, Rust 1.98.1 (the build needs the cargo
registry/vendor state already present under `/scratch/g942/cargo`; this is
unverified).

## 4. Recorded card results

| Run | Measured wall time | Hard limit | Status |
|---|---|---|---|
| quick | 4.239929567 s | `timeout --signal=KILL 600` | 18/18 pass |
| full | 10.782849082 s | `timeout --signal=KILL 600` | 47/47 pass |

These observations belong to the committed receipts and toolchain above; they
are not kernel throughput measurements. A run killed by the timeout is failed,
not a pass. Re-run Tier C when the PM gfx942 code-object bytes change.

## 5. Receipts

Quick and full runs write **separate** receipts and must never share a path:

- quick: `/scratch/g942/pmcdna/<sha>/cdna_card_tierc.quick.json`
- full: `/scratch/g942/pmcdna/<sha>/cdna_card_tierc.json`

Copy both back for review:

```sh
scp <user>@<card-host>:/scratch/g942/pmcdna/<sha>/cdna_card_tierc{.quick,}.json \
    <user>@<card-host>:/scratch/g942/pmcdna/<sha>/toolchain.txt .
```

A quick pass says nothing about the full run, and the reverse.

### What a receipt records

One JSON file per run, schema `cdna_card_tierc/1`, rewritten after every case
(including cases with failed comparisons), so a killed run leaves a partial
receipt with `"complete": false`. Top-level keys: `schema`, `mode`
(`"card"` or `"dry-run"`), `quick`, `args`, `host`, `tools` (`hipcc` and `clang`
`--version` output), `runtime` (`device_count`, `device`, `device_arch`,
`hip_runtime_version`; card mode only), `assumptions` (the
`[UNVERIFIED: confirm on MI300X]` items), `default_accum_model`,
`code_object_version`, `poison_byte`, `kernels` (PM code-object bytes, size and
SHA-256, symbols), `hsaco_dir`, `hsaco` (per twin: path, bytes, SHA-256, or the
read error), `cases`, `errors`, `summary` (`cases`, `pass`, `fail`,
`setup_or_run_errors`), `wall_seconds`, `complete`, `exit_ok`.

Each entry of `cases` has `kernel`, `case`, `seed`, `shape`, `grid`, `block`,
`buffers`, `status` (`PASS`/`FAIL`), the four comparison results below,
`hip_emulation` (`available`, `reason` when not), `card_columns`
(`"hardware"` or `"dry-run: second emulator run"`), `cardHIP_source`,
`models`, `model_matches`, `outputs`, `errors`, `wall_seconds`. Each element of
`outputs` is one output buffer compared separately, with per-comparison counts
(`mismatch_elements`, `mismatch_bytes`, `max_ulp_finite` for f32), the first 8
mismatches (`first_mismatches`) and SHA-256 of each execution.

### Comparisons

Names: **cardPM** = the PeaceMaker-emitted code object on the card; **cardHIP**
= the HIP twin on the card; **cpu_ref** = the CPU reference of the kernel
module; **simPM** = the emulator running the PeaceMaker code object; **simHIP**
= the emulator running the HIP twin's code object.

| Key | Comparison | Gates exit status |
|---|---|---|
| `a_cardPM_vs_cardHIP` | byte identity | yes |
| `b_default_cardPM_vs_cpu_ref` | default model | yes |
| `c_default_cardPM_vs_simPM` | default model | yes |
| `d_cardHIP_vs_simHIP` | recorded; `null` when the emulator cannot run the twin (reason in `hip_emulation`) | no |

Every output buffer is compared by bits. The process exits nonzero if any case
has an (a), (b-default) or (c-default) mismatch, an execution error, or a setup
failure.

### Accumulation models

Only `fp8_moe_gate_up` depends on the MFMA accumulation model (QSA decode does
not use MFMA). For its (b) and (c), the harness evaluates `Cdna3Fused`
(the default, `Config::default()`), `Exact`, `Serial`, `Grouped(4)`,
`Grouped(8)`, `Grouped(16)`, `Grouped(32)` and lists each in `model_matches`
(`b_cardPM_matches_cpu_ref`, `c_cardPM_matches_simPM`, `default`).

- **Only the default model, `Cdna3Fused`, decides the verdict.** It is still
  `[UNVERIFIED: confirm on MI300X]` in source; the card run tests it.
- The other models are probes. A matching non-default model never masks a
  default failure, and never changes the gate or the emulator default.
- A match shows the emulator under that model reproduces the card on these
  cases, not that the model equals the silicon in general. Changing the default
  is a separate decision made from real-hardware receipts.

## 6. Reading the result

A run is a pass only if all of these hold:

1. The process exited 0 (the receipt has `"complete": true`, `"exit_ok": true`
   and `summary.fail == 0`, `summary.setup_or_run_errors == 0`).
2. `mode` is `"card"`, never `"dry-run"`, and each case has
   `"card_columns": "hardware"` and `"cardHIP_source": "hardware HIP twin"`.
3. `runtime.device_arch` is `gfx942`.
4. This holds for the quick receipt and, separately, for the full receipt.

Then read `model_matches` on the `fp8_moe_gate_up` cases to see which
accumulation models matched bit for bit. Anything else (timeout, build failure,
missing receipt, any failed case) is not a pass; report it with `toolchain.txt`
and the receipt that was written.
