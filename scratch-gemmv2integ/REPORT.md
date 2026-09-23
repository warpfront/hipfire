# GEMM v2 V2C production integration: gfx1100

Branch `gfx11-gemmv2`, based on `land-042` at `776358d6d`, in hipx worktree `/home/kaden/hipfire-gemmv2`. Model: `/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`. Standalone source and oracle: GemmV2Gfx11's `/home/kaden/hipfire-prof040/scratch-gemmv2/` (frozen `gemmv2.hip` sha `621ac156…`, V2C-ms = `-DV2_TN=128 -DV2_TM=128 -DV2_MAGIC=1 -DV2_SCF32=1`).

**Decision:** on exact gfx1100, route symmetric MQ4V2 IU4 full-tile SET and ADD prefill GEMMs through V2C. Every gfx1100 gate below passes. On XTX, pp8192 goes from **2,334.7 to 2,564.9 tok/s (+9.86%)**, which clears the 2,500 target. gfx1151 is unchanged and still dispatches X5, pending GemmV2Gfx11's Halo verdict and Main's approval.

## Layout: the artifact and producer are unchanged, and nothing is repacked

V2C as measured consumes NativeIu4SoA1 (DESIGN §8.3). Neither production option for supplying that layout was usable:
- **Load-time W sidecar:** about 0.516 × MK bytes, roughly 14 GB for this model, alongside the 15 GB artifact. It does not fit in the XTX's 24 GB.
- **Per-call W repack:** about 90 MB of traffic per GEMM, roughly 0.1 ms. That is about 16% of an N=512 GEMM.

With Main's approval, `kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip` instead keeps V2C-ms's compute unchanged:
- the M128×N128 8-wave tile, 2 row × 4 token fragments per wave, and two K128 LDS slots with one barrier per epoch;
- the MAGIC integer chains, SCF32 fold, epilogue and ADD grouping.

It stages the production operands directly:
- **Weights:** MQ4G256V2 136-byte groups.
- **Activations:** `block_i4_128` 72-byte blocks.
- **Permutation:** the staging lanes apply the NativeIu4SoA1 fragment-major permutation at the LDS store. Weights are rebiased with `^0x88888888` during staging.

Staging lane (hi, rr) of wave w owns row/token `16w+rr` and K16 slices `2i+hi`, so each b64 store instruction fills one contiguous 256-byte LDS block with no bank conflicts. Scales do not use LDS: a third LDS region pushes past 32 KiB and drops the WGP to 1 CTA.
- **W scales:** each lane loads one row's FP16 scale, and the fold gathers the rows it needs with DPP `row_share`.
- **X scales (`d`):** read straight from `Xq`.

ABI: `(const char* A, const char* Xq, float* Y, int M, int K, int N)`. Symbols `gemm_mq4g256v2_residual_iu4_v2c_{set,add}_gfx11`. Grid `(N/128, M/128)`, block `[32,8,1]`, dynamic LDS 32,768 B.

## Dispatch

`crates/rdna-compute/src/gemm.rs` (`gemm_mq4g256v2_mmq_prequant_iu4`) selects V2C using the X5 eligibility rule, restricted to exact gfx1100. The conditions are:
- symfold is active (symmetric artifact, `gfx11_iu4_symfold`);
- `M%128 == N%128 == 0`, with `K%256 == 0` as the existing precondition;
- dimensions are positive;
- the eager column route is in use (not capture or replay);
- `arch == "gfx1100"`.

V2C does not cover more shapes than X5, so the rule was not widened. It does not depend on the X5 shape knob.

`HIPFIRE_IU4_V2C=0` falls back to X5. Smoke test with the knob set (`gfx1100/killswitch.log`): the fresh kernel cache contained only `…_gfx11_x5_symfold.936038d287cc3b6f.hsaco` and no V2C object, and pp512 ran at 2,164 tok/s in one uncontrolled run.

`kernels.rs` registers `GEMM_MQ4G256V2_RESIDUAL_IU4_V2C_GFX11_SRC` as a separate module. No existing translation unit or compiler policy changed.

## Standalone comparison (gfx1100, N=8192)

These runs used GemmV2Gfx11's host with one extra arm (`standalone/gemmv2-host.cpp`, `run_standalone.sh`, `summ.py`), on the non-periodic oracle corpus.
- **Oracle** (`logs/S5-oracle.log`): the production-layout V2C is bit-exact against pinned X5 on SET and ADD. 0 of 7,340,032 outputs differ, covering K = 256, 512, 5120, 6144 and 17408, the witness/extrema/scales/one-hot corpora, and multi-tile grids. Native V2C-ms also shows 0 differences.
- **Timing:** each value is the mean of a forward-order and a reverse-order run, each the median of 7. The percentage is of the 253.296 TOPS XTX iu4 reference.

| N8192 shape | X5 ms / TOPS / % ceil | V2C-ms native ms / TOPS / % ceil | V2C prod-layout ms / TOPS / % ceil | prod speedup vs X5 | native speedup vs prod |
|---|---:|---:|---:|---:|---:|
| set_m17408 | 9.802 / 149.0 / 58.8% | 8.075 / 180.8 / 71.4% | 8.619 / 169.4 / 66.9% | 1.137x | 1.067x |
| set_m10240 | 5.799 / 148.1 / 58.5% | 4.763 / 180.3 / 71.2% | 5.074 / 169.3 / 66.8% | 1.143x | 1.065x |
| set_m6144 | 3.499 / 147.3 / 58.2% | 2.885 / 178.6 / 70.5% | 3.069 / 167.9 / 66.3% | 1.140x | 1.064x |
| add_k17408 | 9.851 / 148.2 / 58.5% | 8.207 / 177.9 / 70.2% | 8.700 / 167.8 / 66.3% | 1.132x | 1.060x |
| add_k6144 | 3.538 / 145.7 / 57.5% | 3.096 / 166.5 / 65.7% | 3.403 / 151.5 / 59.8% | 1.040x | 1.099x |

Objects compared (`logs/S5-objects.sha256`):
- X5 `fc4ecc4d…` (`hipfire-x5` runtime cache);
- native `v2c-ms-gfx1100.hsaco` `8700df72…`;
- production-layout `a8319e40…`.

**What a native file layout (V3) would still buy:** about 6–7% GEMM time on SET and ADD K17408, and about 10% on ADD K6144.

The ablations below use diagnostic arms that produce wrong answers (`standalone/v2cp_diag*.hip`, `logs/D-*`). The 6–7% gap on SET and ADD K17408 breaks down roughly as:
- about 2% from the 72-byte-strided `d` loads;
- about 1.3% from scattered staging addresses;
- the remainder from issuing twice as many staging loads (b64 instead of b128).

Development steps, with SET M17408 TOPS:

| Variant | SET M17408 TOPS | Notes |
|---|---:|---|
| W scales staged in LDS | 157 | 33,280 B LDS, which gives 1 CTA/MP |
| `ds_bpermute` scale gather | 152 | `lgkmcnt(0)` serialization |
| `readlane` + `cndmask` | 159 | |
| Conflict-free store map + DPP `row_share` (shipped) | 169 | |
| Misaligned b128 staging (`v2cp5`) | 167 | not shipped |
| `permlanex16` `d` gather (`v2cp6`) | 168 | not shipped |

## Resources: runtime-compiled production HSACO

The object is `trace/gfx1100-8192/home/.hipfire_kernels/gfx1100/gemm_mq4g256v2_residual_iu4_v2c_gfx11.491946f1066261ce.hsaco` (sha `1e0e06aa…`), compiled with `--genco --offload-arch=gfx1100 -O3 --no-offload-compress`. Probe output (`resource.cpp`, `gfx1100-resources.txt`):

| Resource | SET | ADD |
|---|---:|---|
| VGPRs | 179 | 179 |
| Private bytes | 0 | 0 |
| VGPR/SGPR spills | 0 / 0 | 0 / 0 |
| Dynamic LDS | 32,768 B | 32,768 B |
| CTAs per MP (waves) | 2 (16) | 2 (16) |

These match the standalone numbers. The disassembly is identical, 0 diff lines, to the standalone-timed `v2cp4` object.

## Gates (gfx1100, ROCR=0 HIP=0, knob-free, VMM q8)

### 1. rocprof pp8192 trace

Script: `run_profile.py` + `inventory.py`. The trace was an uncached second request after a warm-up request. Asserted: `GPU dev 0: gfx1100`, `KV cache: Q8 vmm (`, and both visibility variables on the server and the daemon.

| Kernel | Calls | ms |
|---|---:|---:|
| V2C SET `gemm_mq4g256v2_residual_iu4_v2c_set_gfx11` | 544 | 1,590.180 |
| V2C ADD `gemm_mq4g256v2_residual_iu4_v2c_add_gfx11` | 256 | 765.390 |
| **SET + ADD** | | **2,355.570** |
| X5 SET + ADD (reference) | | 2,667.336 |

The call counts equal X5's, so every X5 call site now dispatches V2C.
- **Saved:** 311.8 ms (−11.69%) against X5.
- **Trace totals:** kernel sum 3,169.4 ms; request wall 3,233.9 ms.

### 2. Fresh-process ABBA ×2

Script: `run_abba.py` against pinned byte copies of the `land-042` release binaries (`baseline-land042/`: `hipfire` `f1ae6d2e…`, `daemon` `a4f83b5c…`, identical to `/home/kaden/hipfire-land042/target/release`). Command, run with a fresh HOME per run and `HIPFIRE_*` scrubbed:

`hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode q8 --kv-backend vmm --json`

Each run asserted its arch, Q8/VMM, and process visibility. Per-process medians in tok/s, in run order:

| # | Arm | pp512 | pp8192 | tg128@128 |
|---:|---|---:|---:|---:|
| 1 | A land-042 | 2281.7 | 2343.9 | 49.031 |
| 2 | B V2C | 2562.9 | 2568.0 | 48.813 |
| 3 | B V2C | 2558.7 | 2564.8 | 48.900 |
| 4 | A land-042 | 2258.1 | 2331.1 | 49.018 |
| 5 | A land-042 | 2262.2 | 2335.1 | 48.947 |
| 6 | B V2C | 2554.6 | 2564.3 | 48.939 |
| 7 | B V2C | 2556.7 | 2562.5 | 49.053 |
| 8 | A land-042 | 2257.0 | 2328.8 | 48.953 |

| Row | land-042 mean | V2C mean | Pooled | Per cycle |
|---|---:|---:|---:|---|
| pp512 | 2264.75 | 2558.23 | **1.12958×** | 1.12816, 1.13102 |
| pp8192 | 2334.73 | **2564.90** | **1.09859×** | 1.09793, 1.09925 |
| tg128 decode | 48.987 | 48.926 | 0.99876× (−0.12%) | 0.99657, 1.00095 |

**PASS:**
- pp8192 is ≥ +1.5%.
- No row is more than 1% slower.
- Decode is within 1%.

Absolute pp8192 on XTX is **2,564.9 tok/s** (lowest V2C process 2,562.5). Raw data: `gfx1100/abba/summary.json`, the per-run `*.json` and `*.log` (force-added, since the repo ignores `*.log`), and `gfx1100-abba.out`.

### 3. Quality

**WT2** (`run_wt2.py`, using this tree's `eval_hipfire`):
- Recipe: `--kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 24`.
- Result: 24,552 tokens, slice-mean KLD **0.076879**.
- The KLD sequence is **byte-identical** to land-042's recorded `B.kldseq`, SHA256 **`8f2b94bb904603bc53d17f7877c306ee4aacd0afc7c71acbf80a60bea0b1e2b2`**.
- The eval kernel cache contains the V2C object, so V2C is on this path.

**Battery** (`run_battery.py`):
- Manual `hipfire serve` with a fresh HOME. Asserted `GPU dev 0: gfx1100`, effective K=q8/V=q8, and ROCR=0/HIP=0 on the server and the daemon (`serve-env.txt`).
- Harness: `scripts/serve_harness.py --no-spawn --mode battery --max-think-tokens 1 --max-tokens 512`.
- Result: **5/5** `finish=stop` on the code, reasoning, factual, prose and instruct prompts. Runaway, empty, attractor and retrieval-miss counts are all 0 (`gfx1100/battery.json`).

**TTFT-511** (`odd_fill.py`): 511 prompt tokens, `finish=stop`, and a coherent answer that recognizes the truncated message and the `merge_sorted` function (`gfx1100/ttft511.json`). N=511 goes through the partial-N route, not V2C.

## Build and provenance

- The release binaries were built **locally** on hiptrx, with the same Ubuntu 26.04.1, glibc 2.43, rustc 1.98.1 and `/opt/rocm` as hipx:
  - `cargo build --release` for the workspace;
  - `cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`.
- The binaries were copied to `/home/kaden/hipfire-gemmv2/target/release` (`hipfire` `afbdf65c…`, `daemon` `2c84b17f…`, `eval_hipfire` `930c7dcf…`). No compile ran on hipx during any peer's timing.
- Kernels are JIT-compiled on first use with the runtime flags above.
- GPU leases were announced on hub. The final `rocm-smi --showpids` showed no gfx1100 process of mine.
- Clean-room check: the assignment's case-insensitive `git log -p land-042..HEAD` grep for the excluded token returns 0.

## Not done / next

- gfx1151: the kernel TU also builds for `__gfx1151__`, but dispatch is gated to gfx1100. Extending it waits for GemmV2Gfx11's Halo verdict and Main's go-ahead.
- `V2_RPF` residual prefetch: GemmV2Gfx11's best native ADD arm (235 VGPR) was not ported. It is a follow-up candidate for ADD K6144.
