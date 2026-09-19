# Halo XDNA2 NPU row-slice screen for prefill GEMMs

Date: 2026-09-16. **Planning-only / read-only screen** for `/home/kaden/ClaudeCode/warpfront/wt-lloyd` (`mq4-lloyd`). No source edits, no GPU/NPU runs, no formatters, no suites, no commit. Parent runs any hardware probe on **hipx** (Linux kernel 7.0, ROCm 7.15, gfx1151 iGPU; NPU node expected at `/dev/accel/accel0` via `amdxdna`). This document packages the software stack, compute facts, runnable probe recipe, hipfire integration sketch, and kill criteria so the parent can execute once without redesign thrash.

> Prior disposition: `docs/plans/2026-09-15-halo-gap-plan.md` listed **NPU** under deferred/rejected levers. This screen does **not** reopen a backend. It defines measurements that either kill the idea cleanly or unlock a separate Sol spec. Do not treat any TOPS or pp2048 gain below as measured on hipx.

## Ten-line summary

1. Stack on kernel 7.x: in-tree or DKMS `amdxdna` + XRT NPU packages (`libxrt-npu2` / `ppa:amd-team/xrt`) + IRON/mlir-aie wheels + Peano (`llvm-aie`); verify with `xrt-smi examine` and `/dev/accel/accel0`.
2. Strix Halo XDNA2 is a **4×8** AIE2P array (32 CompTiles), marketed **50 TOPS int8**; L1 64 KiB/tile, L2 512 KiB/MemTile; effective NPU↔DRAM in published GEMM microbench **~50 GB/s** (shared LPDDR with iGPU/CPU).
3. Native matmul dtypes: **int8×int8→int32** (and narrowed out), **bf16** (AIE2P often via **bfp16**), hardware **bfp16**; public **int4 weight** paths are fused dequant research, not a free IU4 twin.
4. Honest sustained int8 GEMM: open mlir-aie whole-array ~**6–8 TOPS**; carefully tuned IRON/chess designs up to ~**36–38 TOPS**; treat **15–30 TOPS** as the practical band for a hipfire-shaped backend.
5. Probe: mlir-aie `whole_array` int8 near **17408×5120×512** and **5120×17408×512** (or largest legal tile multiple); report µs and TOPS; then concurrent NPU matmul vs `tmp_halo_iu4_oracle` `TIME=1`.
6. Integration sketch: **row-split M** (each output row owned by exactly one device); bit-exact per row only if NPU **int32 accumulate + IU4_FOLD_RN-class f32 fold** match the GPU consumer.
7. NPU **can** int32 MAC accumulate; **cannot** silently match GPU f32 fold with fp16 weight headers without an explicit fold kernel or host epilogue — that is the numerical gate, not a free lunch.
8. Weights: MQ4 nibbles need **int8 expansion (2× payload)** or on-tile unpack; activations `block_i4_128` → int8 (or custom int4 design); XRT↔HIP sync is **host wait only** (no cross-device fence).
9. **Kill:** sustained probe TOPS **< 15**, or concurrent GPU GEMM slowdown **> 20%** from DRAM contention.
10. **If kills pass:** projected pp2048 gain on Qwen3.8-27B IU4 path is roughly **+5% … +18% tok/s** (conservative half-M offload, 15–25 TOPS, ≤20% contention) — not a free 2×; reopen only with a Sol spec.

---

## 1. Software stack on Linux (Strix Halo / Ryzen AI MAX 395, XDNA2)

### 1.1 Kernel driver: `amdxdna`

| Fact | Source |
|---|---|
| In-tree driver under `drivers/accel/amdxdna/` | [torvalds/linux `drivers/accel/amdxdna`](https://github.com/torvalds/linux/tree/master/drivers/accel/amdxdna) |
| Merged mainline in **Linux 6.14** (DRM/accel) | [Phoronix Linux 6.14 DRM pull](https://phoronix.com/news/Linux-6.14-DRM-Feature-Pull); [amd/xdna-driver README](https://github.com/amd/xdna-driver/blob/main/README.md) (“Ubuntu 25.04 includes Linux kernel 6.14 … amdxdna”) |
| Minimum practical base: kernel **≥ 6.10** (AMD IOMMU SVA + `CONFIG_DRM_ACCEL`) | [amd/xdna-driver README — System Requirements](https://github.com/amd/xdna-driver/blob/main/README.md) |
| Strix / Halo PCI ID **`1022:17f0`** | xdna-driver + community XDNA2 notes |
| Device node | **`/dev/accel/accel0`** (accel subsystem), group `render` |
| Firmware | `/lib/firmware/amdnpu/` (e.g. `17f0_10/` / `17f0_11/`); protocol must match driver |
| hipx context | Kernel **7.0** already exceeds 6.14 — in-tree `amdxdna` is expected; still match firmware to userspace |

**Honest caveat:** in-tree lags AMD’s out-of-tree DKMS for protocol/firmware features. For LLM-class BOs and recent XRT SHIM, DKMS from `amd/xdna-driver` or the AMD XRT PPA is still the common path even on 7.x ([xdna-driver issues on firmware mismatch](https://github.com/amd/xdna-driver/issues/1219)).

### 1.2 Userspace: XRT + xrt-npu

XRT (Xilinx Runtime) + the XDNA SHIM/plugin provide `xrt-smi`, `libxrt`, `pyxrt`, and the NPU context path.

**Preferred on Ubuntu with kernel ≥ 6.17 (and fine on 7.x)** — from [Xilinx/mlir-aie README](https://github.com/Xilinx/mlir-aie/blob/main/README.md):

```bash
# BIOS: latest firmware; Secure Boot off if using unsigned DKMS
# Kernel check
uname -r   # expect 6.17+ / 7.x on hipx

# AMD XRT + NPU packages
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update
sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms
sudo usermod -aG render "$USER"
# memlock: required for large locked BOs (default 8 MiB breaks 64 MiB mmap)
sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null <<'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF
# On systemd user sessions also raise user@.service LimitMEMLOCK (see community GOTCHAS)
sudo reboot
```

**From-source alternative** ([amd/xdna-driver](https://github.com/amd/xdna-driver)):

```bash
sudo apt-get install git-lfs && git lfs install
git clone https://github.com/amd/xdna-driver.git && cd xdna-driver
git submodule update --init --recursive
sudo ./tools/amdxdna_deps.sh
cd xrt/build && ./build.sh -npu -opt && cd ../../build && ./build.sh -release
# install generated Release/xrt_plugin.*-amdxdna.deb (and base XRT debs)
source /opt/xilinx/xrt/setup.sh
```

Ubuntu 26.04+ may ship `libxrt-npu2` natively ([Launchpad `libxrt-npu2`](https://launchpad.net/ubuntu/+source/xrt)); still verify firmware/driver pairing.

### 1.3 IRON / mlir-aie / Peano

| Component | Role | Canonical repo |
|---|---|---|
| **mlir-aie** | MLIR AIE dialect + codegen | [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) |
| **IRON** | Close-to-metal Python API (`aie.iron`) | same repo (`import aie.iron as iron`) |
| **Peano** | Open LLVM/Clang AIE backend (`llvm-aie` wheels) | [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) |
| Optional **xchesscc** | Proprietary single-core compiler (Vitis AIE Essentials) | Ryzen AI EA lounge — **not required** for Peano path |

**Install (wheels — fastest path, from mlir-aie README):**

```bash
git clone https://github.com/Xilinx/mlir-aie.git && cd mlir-aie
# pin a release tag if reproducing; wheels must match tree commit
sudo apt install build-essential clang clang-14 lld lld-14 cmake ninja-build \
  python3-venv python3-pip uuid-dev
source utils/env_install.sh          # creates ironenv + mlir_aie + peano pins
source utils/env_setup.sh            # every new shell
```

Manual equivalent: `pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/<tag>` and `pip install -r utils/peano-requirements.txt`.

Device naming decoder (do not mix layers):

| Layer | Strix Point | Strix Halo | Source |
|---|---|---|---|
| lspci | `1022:17f0` NPU | same family | hardware |
| XRT name | `RyzenAI-npu4` | **`RyzenAI-npu5`** (typical) | [xdna-driver](https://github.com/amd/xdna-driver); [ryzen-npu-linux XDNA2.md](https://github.com/Jonas-Augustinus-Linus/ryzen-npu-linux/blob/main/docs/XDNA2.md) |
| mlir-aie / IRON | `npu2` | `npu2` (8-col) | mlir-aie Devices / examples |
| IREE amd-aie | `npu4` | Halo often still experimental vs Point | iree-amd-aie |

### 1.4 Verify (must pass before any probe)

```bash
source /opt/xilinx/xrt/setup.sh 2>/dev/null || true
lsmod | grep amdxdna || test -d /sys/module/amdxdna
ls -l /dev/accel/accel0
groups | grep -q render && echo "render: ok"
ulimit -l   # expect unlimited or very large
xrt-smi examine
xrt-smi validate   # if packaged
python3 -c 'import pyxrt; d=pyxrt.device(0); print(d.get_info(pyxrt.xrt_info_device.name))'
```

**Pass bar:** `xrt-smi examine` lists an NPU (e.g. `RyzenAI-npu5` / Strix Halo), `/dev/accel/accel0` is RW for the user, `pyxrt` opens device 0, memlock is not 8192 KiB.

---

## 2. NPU compute facts (XDNA2 / Strix Halo)

Primary architecture + measured GEMM source: Taka et al., *Striking the Balance: GEMM Performance Optimization Across Generations of Ryzen AI NPUs*, arXiv:2512.13282 (Dec 2025) — [HTML](https://arxiv.org/html/2512.13282v1) / [PDF](https://arxiv.org/pdf/2512.13282). Array geometry also in [mlir-aie whole_array README](https://github.com/Xilinx/mlir-aie/blob/main/programming_examples/basic/matrix_multiplication/whole_array/README.md) and [ryzen-npu-linux XDNA2.md](https://github.com/Jonas-Augustinus-Linus/ryzen-npu-linux/blob/main/docs/XDNA2.md).

### 2.1 Array and memory

| Item | XDNA2 value | Notes |
|---|---|---|
| CompTiles | **4 rows × 8 columns = 32** | Full array usable for GEMM (asymmetric 4×8 map) |
| MemTiles / ShimTiles | 8 + 8 (one column stack) | L2 row under compute; shim to DRAM |
| L1 per CompTile | **64 KiB** data + 16 KiB program | Banked scratchpad |
| L2 per MemTile | **512 KiB** | Neighbor spill/access on XDNA2 |
| Marketed peak | **50 TOPS int8** (dense) | Also ~50 TOPS-class bfp16 marketing on some slides |
| Clock (paper turbo) | up to **~1.8 GHz** XDNA2 | vs ~1 GHz XDNA1 |
| DMA channels | Comp/Shim: 2 MM2S + 2 S2MM; Mem: 6+6 | 3D/4D BD addressing; on-the-fly layout transform |
| Effective NPU↔DRAM (GEMM-like microbench) | **~50 GB/s** XDNA2 | Paper dual-channel DDR5-5600 mini-PC; **not** full Halo LPDDR peak |
| System DRAM (Halo) | LPDDR5X, often **~200+ GB/s** theoretical shared | CPU + **gfx1151 iGPU** + NPU contend |

### 2.2 Matmul datatypes

| Path | Support | Notes for hipfire |
|---|---|---|
| **int8 × int8 → int32** | Native, first-class | Primary probe dtype; AIE mmul modes (e.g. 8×8×8 on npu2) |
| int8 → int16 / int8 out | Native (precision reduction) | Higher TOPS, not IU4 fold compatible |
| **bf16** | Yes; on AIE2P often **emulated via bfp16** | Peak bf16 end-to-end ~**14.7 TOPS** (paper); open stack uses `--emulate-bf16-mmul-with-bfp16` |
| **bfp16** (block FP, 8 vals / shared exp) | **Hardware on XDNA2** | Good for some ML paths; not MQ4 IU4 |
| **int4 weights** | Not a free ISA twin of `wmma_iu4` | Public work = fused dequant+GEMM (TileFuse / community W4A16 ~6 TOPS class) or host expand to int8 |
| FP32 MAC array-wide | Not the NPU’s job | Fold/epilogue only |

### 2.3 Honest sustained int8 TOPS

| Class | int8 TOPS | Context |
|---|---:|---|
| Marketing peak | 50 | Ideal, dense, turbo |
| Paper end-to-end optimized (chess/AIE API, balanced tiles) | **up to 38.05** (typ. high 30s @ ~4K) | arXiv:2512.13282 Table 3: e.g. 37.35 TOPS at 4032×4320×4608 int8→int8 |
| Single-core efficiency | 384–450.6 MAC/cycle | Same paper Table 1 |
| Open mlir-aie `whole_array` (Peano, community Strix Point) | **~6.65** at 2048³ i8 | [ryzen-npu-linux XDNA2.md](https://github.com/Jonas-Augustinus-Linus/ryzen-npu-linux/blob/main/docs/XDNA2.md) verified |
| Community W4A16 fused | ~5.9–9 | Different problem (dequant+GEMM) |

**Sustained range to plan against:** **~6 TOPS (stock example) … ~30–38 TOPS (hero kernel)**. For kill criteria we use **15 TOPS** as “worth a backend” floor — above stock examples, below paper hero, reachable only with real tiling/DMA work.

---

## 3. Probe recipe (parent runs on hipx — no execution in this screen)

### 3.0 Preconditions

1. §1.4 verify green.
2. `source` ironenv + `utils/env_setup.sh` + XRT setup.
3. iGPU left available for concurrency leg (`ROCR_VISIBLE_DEVICES` as used for Halo IU4 oracles).
4. Record: `uname -r`, `xrt-smi examine` full dump, NPU FW version, `rocminfo` gfx1151 name, git SHAs of mlir-aie + peano wheels.

### 3.1 Solo int8 matmul (TOPS + µs)

Use the canonical whole-array example:

```bash
cd /path/to/mlir-aie
source utils/env_setup.sh
source /opt/xilinx/xrt/setup.sh

cd programming_examples/basic/matrix_multiplication/whole_array

# Sanity: default-ish square (document actual flags from --help on the pinned tree)
python3 whole_array.py --dtype_in i8 --dtype_out i32 --dev npu2 \
  -M 2048 -K 2048 -N 2048
# Expect: PASS vs numpy/int ref; print or wrap timing.

# Hipfire-shaped large M (gate/up-ish): M=17408, K=5120, N=512
# Dimensions MUST be multiples of the design's m,k,n tile and n_aie_cols=8.
# If illegal, round each dim UP to the next legal multiple and report the
# rounded shape + padded TOPS separately from "payload TOPS".
python3 whole_array.py --dtype_in i8 --dtype_out i32 --dev npu2 \
  -M 17408 -K 5120 -N 512   # adjust flags to tree's CLI

# Transpose-ish / down-ish: M=5120, K=17408, N=512
python3 whole_array.py --dtype_in i8 --dtype_out i32 --dev npu2 \
  -M 5120 -K 17408 -N 512
```

Makefile path (if preferred on the pinned tree):

```bash
# Illustrative — confirm variable names in makefile-common on the checkout
make clean
make dtype_in=i8 dtype_out=i32 devicename=npu2 n_aie_cols=8 \
     M=2048 K=2048 N=2048 m=64 k=64 n=64
make run
```

**Timing harness (required metrics):**

- Warmup ≥10; sample ≥50 steady iterations; report **median and p90 wall µs** (host chrono around XRT run, or example’s own timer if documented).
- TOPS formula: `TOPS = 2 * M * K * N / (t_seconds * 1e12)`.
- Also log: tile `(m,k,n)`, `n_aie_cols`, dtype, xclbin build id, whether Peano or chess.
- If 17408×5120×512 will not place/compile, binary-search largest **M** multiple of `m*4` at fixed K=5120, N=512 (and symmetrically largest K at M=5120, N=512). Record max legal shape.

**Pass interest (not yet kill):** median ≥ **15 TOPS** on at least one hipfire-near shape.

### 3.2 Concurrency / DRAM contention probe

Goal: NPU matmul hammer while iGPU runs the shipping IU4 oracle path; measure **iGPU-only** vs **iGPU+NPU** GEMM time.

**GPU leg** (existing throwaway; lab feature):

```bash
# From wt-lloyd on hipx — parent owns exact model path / DEVICE
HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
  HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-npu-screen HIPFIRE_GFX11_MQ4V2_IU4=1 \
  TIME=1 \
  cargo run --release -p hipfire-runtime --example tmp_halo_iu4_oracle --features lab -- \
    /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt
```

(`TIME=1` is already honored in `crates/hipfire-runtime/examples/tmp_halo_iu4_oracle.rs`.)

**NPU leg:** loop the best solo shape from §3.1 in a tight Python/C++ XRT process (no exit between iters).

**Protocol:**

1. **G0:** GPU oracle alone, 20 timed iters after warmup → median `t_G0` for the hot IU4 symbol(s) or whole-oracle wall if that is what TIME prints.
2. **N0:** NPU alone, same → median `t_N0`, TOPS.
3. **C:** start NPU loop; after it is hot, run GPU oracle identical to G0 → median `t_G|N`; optionally sample NPU TOPS under load `T_N|G`.
4. Contention metric: `slowdown = (t_G|N - t_G0) / t_G0`.
5. Optional reverse: GPU loop + timed NPU batch.

Pin affinity lightly if needed so the NPU host thread does not thrash the GPU submitter; do **not** change GPU clocks mid-run. One power mode (NPU turbo if `xrt-smi` exposes it; iGPU as in IU4 oracles).

**Kill:** `slowdown > 0.20` on the GPU GEMM median.

### 3.3 Log template (paste into parent notes)

```
date / host / uname -r / xrt-smi examine (NPU name, FW)
mlir-aie tag / peano version
shape M K N / tiles m k n / cols / dtype
solo: median_us p90_us TOPS
gpu G0 median_us
concurrent G|N median_us slowdown
N under load TOPS (optional)
kill_tops: PASS|FAIL
kill_dram: PASS|FAIL
```

---

## 4. Integration sketch for hipfire (only if kills pass)

### 4.1 Work split: row-slice M

Prefill IU4 consumers are large **M×K×N** GEMMs with M ∈ {gate/up 17408, down 5120, qkv pieces, …}, K model dim or intermediate, N = prefill chunk (512 today).

**Rule:** partition **output rows** so each row of Y is computed by **exactly one** of {iGPU, NPU}.

- iGPU: rows `[0, M_gpu)` via existing `gemm_mq4g256v2_residual_mmq_iu4_*`.
- NPU: rows `[M_gpu, M)` via XRT graph.
- No split-K across devices (would break ownership and need partial-sum reduce in f32 with ordering headaches).

Bit-exact **per row** is possible in principle because rows do not mix. Cross-device bitwise equality to a full-GPU baseline still requires the NPU path to reproduce the GPU numeric contract on those rows.

### 4.2 Numeric contract: what IU4 actually does

Shipping consumer (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`):

- Weights: **MQ4 nibbles** fed to `wmma_i32_16x16x16_iu4` (no int8 expand in loader).
- Activations: **`block_i4_128`** (72 B: `float d`, `int s`, 64 B nibbles) per 128-K group.
- Accumulate: **int32** WMMA chain over K.
- Fold (one per 128-K half), schematic:

```text
f += sc_w * d_x * C + zp_w * d_x * s_x
```

with fp16/f32 weight scales/zero-points from MQ4 headers and `d_x` / `s_x` from the activation block (`s_x` = exact int sum of the 128 quantised x). That fold is the **IU4_FOLD_RN** family contract the oracles pin.

| Step | NPU can? | Notes |
|---|---|---|
| int8×int8 → int32 MAC | **Yes** | Native |
| int4×int4 → int32 MAC | **Not as gfx11 IU4 WMMA** | Need expand, custom unpack kernel, or different quant |
| Hold int32 partials across K tiles | **Yes** | Output-stationary map in IRON GEMMs |
| f32 fold with **fp16 weight headers** + `d_x`/`s_x` | **Only with explicit kernel/host math** | Not automatic in stock int8 matmul examples |
| Match RN / FMA ordering / flush behavior | **Hard** | Gate with row-wise bit-exact vs GPU on oracle shapes; expect **ulp policy** (bit-exact or documented tol) before production |

**Practical backend choices (Sol would pick one):**

1. **NPU int8 GEMM + host/iGPU fold:** NPU returns int32 C tiles; GPU applies IU4 fold. Preserves fold bit-exactness; adds round-trip BW.
2. **NPU fused fold kernel:** reimplement fold in AIE scalar/vector f32; prove vs `tmp_halo_iu4_oracle` rows.
3. **Accept non-bit-exact NPU rows:** quality gate (KLD/eval) — higher bar, probably wrong for hipfire’s IU4 redline culture.

### 4.3 Weight layout

| Approach | DRAM bytes vs MQ4 nibbles | Pros | Cons |
|---|---|---|---|
| Host expand nibbles → int8 once per weight tensor (or per layer touch) | **~2×** payload | Stock int8 matmul examples work | Memory budget on 128 GB Halo still OK for 27B-class; hurts cache/BW |
| On-tile unpack DMA/core | ~1× store + compute | Better BW | Custom IRON kernel; schedule cost |
| Stay on GPU IU4 for weights, NPU only on a different dtype path | — | — | Out of scope for this screen |

MQ4V2 packing stays the GPU source of truth; NPU mirror is a **derived** buffer, prefill-only, discardable.

### 4.4 Activation sharing

GPU path produces `block_i4_128` (C2 producers or `quantize_int4_mmq_ds128`). NPU stock int8 matmul wants plain int8 (or int16) activations.

- **Minimum viable:** host/iGPU expand `block_i4_128` → int8 activations for NPU rows only (still use `d,s` later for fold).
- **Better:** NPU kernel consumes packed int4 + headers (custom) — more work, less expand BW.
- Activations are **row-partitioned** with M; K-group headers must follow the same groups as GPU (128-K).

### 4.5 XRT sync vs HIP streams

There is **no** device-side fence between gfx1151 and the NPU today in hipfire.

```text
HIP:  quant / produce X  →  record event E_x
Host: wait E_x
Host: memcpy or zero-copy publish X_npu, W_npu slices
XRT:  run NPU gemm on row block
HIP:  launch GPU gemm on complementary rows (can overlap NPU after inputs published)
Host: wait XRT completion + HIP event E_gpu
HIP:  optional fold/epilogue / consume Y
```

- Overlap window = after inputs are visible to both.
- Do not pretend `hipEventRecord` waits for NPU.
- Error paths: if NPU fails, full-GPU fallback for that launch (Sol policy).

### 4.6 Memory budget / unified memory

Strix Halo is a **unified memory** APU: NPU Shim DMA and iGPU both pull from the same LPDDR pool.

| Question | Working answer |
|---|---|
| True SMU zero-copy shared HIP↔XRT buffer? | **Do not assume.** XRT BOs are often explicit; some paths allow host-ptr import. Probe with a small BO: allocate HIP, register/import to XRT (or vice versa), checksum. |
| Conservative plan | Separate XRT BOs; `hipMemcpy`/host staging for X and Y row blocks; weights resident in NPU BO for prefill session |
| Extra footprint | int8 weight mirror (~2× nibble region for offloaded layers) + int8 X mirror for NPU rows + int32/f32 Y staging |
| Contention | Same physical DRAM → §3.2 is mandatory before any backend |

### 4.7 Which layers to slice first

Highest IU4 time share (from Halo gap ledger): **gate/up** `full_set` (M=17408,K=5120) and **down** `full_add` (M=5120,K=17408). Row-split those two families first; leave tiny tails and GDN on GPU.

---

## 5. Kill criteria (hard)

| # | Metric | Kill if | Rationale |
|---|---|---|---|
| K1 | Sustained solo NPU int8 TOPS on probe shape(s) | **< 15 TOPS** | Below this, even perfect 50% M split barely beats GPU overhead; stock ~6–7 TOPS examples already fail |
| K2 | GPU GEMM slowdown under concurrent NPU load | **> 20%** | Shared LPDDR; IU4 is BW-sensitive; >20% eats the offload win |
| K3 (soft, Sol) | Row numeric gate | Cannot match fold policy | Blocks production merge even if K1/K2 pass |

If K1 or K2 fails: **close NPU prefill GEMM** again; do not spend Sol time. If both pass: write a dedicated Sol spec (buffer ownership, fold path, fallback, exact gfx1151-only flag).

---

## 6. Projected pp2048 gain **if** kills pass

Denominators (planning, from Halo IU4 campaign docs — not remeasured here):

- IU4 dominates prefill wall (~**75–80%** of profiled pp512/pp2048 on Qwen3.8-27B XT IU4 path).
- Representative per-call times ~1.5–1.7 ms class for large set/add at N=512 after A5; four chunks at pp2048.

**Model (order-of-magnitude, Amdahl-style):**

- Offload fraction of IU4 MACs `f ∈ [0.30, 0.50]` (row share).
- NPU time for its rows ≈ `(f * ops) / TOPS`; must be ≤ GPU time on remaining rows after contention.
- GPU remaining time ≈ `(1-f) * t_IU4 * (1+slowdown)`.
- Wall IU4 ≈ `max(t_NPU, t_GPU_rem) + t_sync`.

| Scenario | TOPS | f | slowdown | IU4 wall factor | pp2048 tok/s delta (IU4 ~78% wall) |
|---|---:|---:|---:|---:|---|
| Barely pass | 15 | 0.35 | 0.20 | ~0.88–0.95 | **~+4% … +9%** |
| Mid | 22 | 0.45 | 0.10 | ~0.75–0.85 | **~+10% … +16%** |
| Strong | 30+ | 0.50 | 0.05 | ~0.55–0.70 | **~+18% … +28%** (optimistic; fold/sync tax often clips this) |

**Planning band to quote:** if K1/K2 pass, expect about **+5% to +18% pp2048 tok/s** on the IU4 27B path after real sync/fold taxes — not a 2× story. Decode (tg) unchanged. Quality unchanged only under bit-exact or approved ulp policy on NPU rows.

Compare: GPU-only IU4 levers (A5, LF, W2) target the same bucket **without** dual-device software; NPU only wins if probes beat those opportunity costs.

---

## 7. Sources (cite list)

1. [amd/xdna-driver README](https://github.com/amd/xdna-driver/blob/main/README.md) — kernel ≥6.10, build/install, `xrt-smi`, DKMS plugin, firmware paths.
2. [Xilinx/mlir-aie README](https://github.com/Xilinx/mlir-aie/blob/main/README.md) — Ubuntu XRT PPA packages, ironenv, Peano wheels, kernel 6.17+ guidance, device + `xrt-smi` verify.
3. [mlir-aie `programming_examples/.../whole_array`](https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/matrix_multiplication/whole_array) — runnable int8/bf16 whole-array GEMM, 4×8 map, ObjectFifo DMA design.
4. Taka et al., arXiv:2512.13282 — XDNA2 **4×8**, 64 KiB L1 / 512 KiB L2, **~50 GB/s** effective DRAM, int8 **≤38.05 TOPS**, bf16 **≤14.71 TOPS**, single-core MAC/cycle tables.
5. [ryzen-npu-linux `docs/XDNA2.md`](https://github.com/Jonas-Augustinus-Linus/ryzen-npu-linux/blob/main/docs/XDNA2.md) — naming decoder (npu4/npu5/npu2), memlock gotchas, open-stack **~6.65 TOPS** i8 whole_array evidence, Halo vs Point notes.
6. [Phoronix: Linux 6.14 DRM](https://phoronix.com/news/Linux-6.14-DRM-Feature-Pull) — amdxdna mainline merge.
7. In-tree hipfire: `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip` (IU4 accumulate+fold); `crates/hipfire-runtime/examples/tmp_halo_iu4_oracle.rs` (`TIME=1`); Halo gap plans for IU4 wall share.

---

## 8. Non-goals / discipline

- No production flag, no `prefill.rs` hook, no weight rewriter in this screen.
- No claim that FastFlowLM/Lemonade proprietary kernels help hipfire’s open IU4 contract.
- No NPU decode story (memory-bound tg; hybrid NPU-prefill + iGPU-decode is industry pattern but out of scope).
- Parent owns hardware; this file is the checklist only.
