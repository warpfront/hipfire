# Halo XDNA2 NPU probe — measured results (hipx)

Date: 2026-09-16. Probe agent: HaloNpuProbe. Recipe:
`docs/plans/2026-09-16-halo-npu-screen.md`. Work area on hipx:
`/home/kaden/npu-screen` (user-space only; no hipfire source touched, nothing
committed). sudo was used for exactly two files (memlock fix, parent-authorized);
all probe runs are user-space.

## 1. What exists on hipx (assignment §1)

- `uname -r`: `7.0.0-31-generic`; Ubuntu 26.04.1 LTS; Ryzen AI MAX+ 395
  w/ Radeon 8060S (MS-S1 MAX, BIOS 1.06); 32 cores; 31211 MB RAM.
- `/dev/accel/`: `accel0` (`crw-rw---- root:render`), driver `amdxdna`
  (in-tree, version `7.0.0-31-generic`) + `amd_pmf`, `gpu_sched`.
- `which xrt-smi`: `/usr/bin/xrt-smi`, XRT **2.21.75** (Debian packages:
  `libxrt2`, `libxrt-npu2`, `libxrt-dev`, `libxrt-utils`,
  `libxrt-utils-npu`, `python3-xrt` — all `1:2.21.75+dfsg-4`). No
  `/opt/xilinx/xrt` (distro packaging, binaries in `/usr/bin`).
- `xrt-smi examine`: NPU **`RyzenAI-npu5` at `[0000:c0:00.1]`**,
  NPU firmware **1.1.2.65**, amdxdna `7.0.0-31-generic`.
- Firmware dir `/lib/firmware/amdnpu/`: `1502_00`, `17f0_10`, `17f0_11`.
- `python3 -c 'import pyxrt; pyxrt.device(0)'` → opens, reports
  `RyzenAI-npu5` (only after the memlock fix below).

### 1.1 Blocker found + fix (memlock)

Initial state: `xrt-smi examine` and `pyxrt.device(0)` both failed with
`mmap(addr=…, len=67108864, prot=3, flags=8209 /* SHARED|FIXED|LOCKED */,
offset=4294967296) failed (err=-11): Resource temporarily unavailable`.
Root cause: `RLIMIT_MEMLOCK` was **8388608 bytes (8 MiB) soft+hard**;
XRT needs a 64 MiB locked mapping. `/etc/security/limits.d/99-amdxdna.conf`
was absent. Raising the hard limit is impossible unprivileged
(`ulimit -l unlimited` and `prlimit --memlock=unlimited` both return
`Operation not permitted` — verified).

Fix applied with parent-authorized `sudo -n` (exact commands):

```bash
echo "* soft memlock unlimited
* hard memlock unlimited" | sudo -n tee /etc/security/limits.d/99-amdxdna.conf
sudo -n mkdir -p /etc/systemd/system/user@1000.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' \
  | sudo -n tee /etc/systemd/system/user@1000.service.d/memlock.conf
sudo -n systemctl daemon-reload
```

Caveat: already-running login sessions keep the old 8 MiB limit (the
`user@1000` manager predates the drop-in; a reboot or
`systemctl restart user@1000.service` would finish the job system-wide).
Working path used for every probe run below: fresh login shells via
`sudo -n -u kaden -i …`, where `ulimit -Sl/-Hl` both report `unlimited`
and `xrt-smi examine` lists the NPU. Nothing else was run with sudo.

## 2. Toolchain (assignment §2, user-space)

- `git clone --depth 1 https://github.com/Xilinx/mlir-aie.git` →
  `/home/kaden/npu-screen/mlir-aie`, main @ `f50bef7` (2026-09-14).
- `utils/env_install.sh` hard-requires a `python3.12` binary; hipx has only
  python3.14, and apt is out of scope — so manual equivalent with python3.14:
  `python3 -m venv ironenv`, `pip install -U -r utils/peano-requirements.txt`
  (pinned `llvm-aie==22.0.0.2026090701+3e93bf7b`), plus the exact matching
  wheel `mlir_aie==1.4.4.dev3+gf50bef7` from `latest-wheels-4`
  (`utils/find_mlir_aie_wheel.py <HEAD>` → `latest-wheels-4
  1.4.4.dev3+gf50bef7`; cp314 assets exist). `import aie` OK.
- Design: `programming_examples/basic/matrix_multiplication/whole_array/whole_array.py`,
  `--dtype_in i8 --dtype_out i32 --dev npu2 --n-aie-cols 8`
  (full 4×8 array), default tiles `m=64, k=64, n=32`, `-w 10 -i 20`.
- Legality (`M%(m*4)==0`, `K%k==0`, `N%(n*8)==0`) holds **exactly** for all
  three shapes below — no padding, payload TOPS == measured TOPS.

## 3. Solo NPU matmul (assignment §2)

FLOPs convention: `2·M·K·N`; TOPS = FLOPs / t(s) / 1e12.
Effective GB/s = int8 A + int8 B + int32 C bytes / NPU time.

| shape (M×K×N) | role | GFOP | NPU avg µs | NPU min / max µs | TOPS (avg/min) | e2e avg µs | GB/s @avg | PASS |
|---|---|---|---|---|---|---|---|---|
| 2048³ | sanity | 17.18 | 4018.3 | 3704.0 / 4307.4 | **4.28 / 4.64** | 4828.5 | 6.26 | PASS (vs numpy) |
| 17408×5120×512 | gate/up | 91.27 | **16099.4** | 15735.6 / 16819.9 | **5.67 / 5.80** | 16976.6 | 7.91 | PASS (vs numpy) |
| 5120×17408×512 | down | 91.27 | **18956.9** | 18476.9 / 19614.8 | **4.81 / 4.94** | 19802.0 | 5.72 | PASS (vs numpy) |

Tiles: `m=64 k=64 n=32`, `n_aie_cols=8` (4×8 full array), Peano (not chess),
dtype i8→i32. Power mode: **Default** (`xrt-smi examine`). GB/s is
`(sizeof(i8)·M·K + sizeof(i8)·K·N + sizeof(i32)·M·N) / t_npu`.
Reference: community stock whole_array i8 ≈ 6.65 TOPS; paper hero
(chess/AIE-API, ~4K) up to ≈ 38 TOPS; kill floor K1 = 15 TOPS (§5).

## 4. Concurrency (assignment §3)

GPU leg (prebuilt, unmodified):
`HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1
HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-lf3 HIPFIRE_GFX11_MQ4V2_IU4=1 TIME=1
/home/kaden/wt-lloyd/target/release/examples/tmp_halo_iu4_oracle
/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt`
(binary + model both confirmed present on hipx).

- G0 (GPU alone): PENDING — runs after all CPU-heavy NPU compiles finish
  (APU power sharing).
- C (GPU + NPU loop): PENDING.
- slowdown = (t_G|N − t_G0)/t_G0; kill K2 = slowdown > 20%.

## 5. Verdict table (per prefill kernel class, chunk-512 layer slice)

GPU baselines per call (pp2048 profile): IU4 gate/up `M=17408,K=5120,N=512`
1533 µs @ 91.27 GFOP (≈59.5 dense-equiv TOPS); IU4 down
`M=5120,K=17408,N=512` 1566 µs (≈58.3 TOPS); qkv/o GEMM 38 µs ×384 calls;
FA2 (H24/KV4/D256, 512 rows × L keys) 1.1 ms @L=2048 (17 ms @32K);
GDN q8 (48 heads, HD128, T=512) 991 µs sequential; small kernels 40–250 µs.

| kernel class | FLOPs/call | bytes/call (NPU i8 path) | NPU feasibility | proj. NPU µs @MEASURED 5.67 TOPS (hero 38) | row-slice helps given contention? |
|---|---|---|---|---|---|
| IU4 gate/up | 91.27 GFOP | A 89.13 MB i8 + B 2.62 MB i8 + C 35.65 MB i32 = 127.4 MB (MQ4 W would be ~44.6 MB nibbles, needs 2× expand) | int8 GEMM native; MQ4 nibbles need int8 expansion; `block_i4_128` X needs expand; IU4_FOLD_RN f32 fold needs explicit kernel/host epilogue | **16 099 measured** (2 402 hero) vs GPU 1533 | See §5.1: at measured TOPS a small row-slice can theoretically cut IU4 wall ~9% with s=0; s≳10% zeros it. Still NO for a backend (K1 FAIL; fold/expand/sync untaxed) |
| IU4 down | 91.27 GFOP | A 89.13 + B 8.91 + C 10.49 = 108.5 MB | same expansion/fold taxes | **18 957 measured** (2 402 hero) vs GPU 1566 | NO — worse TOPS than gate (4.81); same K1 |
| qkv/o GEMM (`gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds`) | ~2.3 GFOP/call at 38 µs GPU ×384 | small, weight-resident | same expansion/fold per call; 384 tiny launches → XRT submit/sync tax dominates | ≫38 µs (launch-bound) | NO |
| FA2 QKᵀ/PV | ~2·512·L·256 per matmul + softmax (L=2048 → ~0.54 GFOP QK + 0.54 GFOP PV) | KV streams L keys | softmax (exp/sum, online renormalize) + causal mask + GQA fanout: no native NPU path in stock flow | n/a | NO |
| GDN q8 recurrence | small GEMMs + scan, 48 heads HD128 T=512 | state carry | strictly sequential T=512 recurrence: AIE array starves; no parallel split | n/a | NO |
| small kernels (rmsnorm/rotate/quant/conv1d/gated_norm/deinterleave) | tiny | streaming, 40–250 µs GPU, BW-bound | NPU slower than launch+XRT round-trip | n/a | NO |

## 6. Kill verdict (§5 criteria)

- **K1 (sustained solo ≥ 15 TOPS): FAIL** — measured stock whole_array int8:
  **5.67 TOPS avg (5.80 best) at the hipfire gate shape 17408×5120×512**;
  4.28 TOPS avg (4.64 best) at 2048³. Even the paper-hero 38 TOPS ceiling
  would give ≈2.40 ms/call vs GPU ≈1.53 ms: the GPU IU4 path (≈59 TOPS
  dense-equiv) is faster than any documented XDNA2 int8 GEMM at these shapes.
- **K2 (contention ≤ 20%): PENDING / informational** — K1 already failed so this
  is not a reopen gate. G0 (GPU oracle alone) then C (NPU hammer + GPU) wait
  on HaloLF7 releasing gfx1151. Stock whole_array hammer is supplemental if a
  later tuned-kernel worker re-measures contention.
- **K3 (row numeric gate): FAIL by inspection** — stock int8→int32 path cannot
  reproduce IU4_FOLD_RN without an explicit fold kernel + bit-exact proof;
  no work spent (correct per screen discipline after K1).

**Disposition: close NPU prefill GEMM.** Do not write a Sol spec. Sustained
open-stack TOPS is **5.67 peak at hipfire shapes, 2.6× below the 15 TOPS
floor**. Even granting the paper-hero 38 TOPS, GPU IU4 (≈59 TOPS dense-equiv)
still wins a full-M race (2.40 ms vs 1.53 ms).

### 5.1 Row-split arithmetic (correction)

A slower NPU can still reduce wall if it runs **in parallel** on a row
fraction `f`, with `wall = max(f·t_N, (1-f)·t_G·(1+s)) + t_sync`.
Using measured gate `t_N=16099 µs`, GPU `t_G=1533 µs`:

| s (GPU slowdown) | optimal f | wall µs | IU4 Δ vs 1533 |
|---|---:|---:|---|
| 0 | 0.087 | 1400 | **−8.7%** |
| 5% | 0.091 | 1464 | −4.5% |
| 10% | 0.095 | 1528 | −0.3% |
| 20% (K2 kill) | 0.102 | 1652 | **+7.8% (loss)** |

So the earlier “any split is a net loss independent of contention” line is
**too strong** at s=0. It is still the right engineering call: the untaxed
9% IU4 win (≈7% pp2048 at 78% IU4 share) is inside noise once expand-to-int8,
host fold, and XRT↔HIP sync are added, and it vanishes by s≈10%. K1 remains
the hard kill.

## 7. Ten-line summary

1. Stack present and FIXED: amdxdna + XRT 2.21.75 + RyzenAI-npu5 FW 1.1.2.65;
   only blocker was memlock 8 MiB → fixed via limits.d + user@ drop-in
   (commands §1.1); probe shells use `sudo -n -u kaden -i`.
2. Toolchain user-space: mlir-aie @f50bef7 + matched wheel
   1.4.4.dev3+gf50bef7 + Peano 22.0.0.2026090701; whole_array i8→i32, npu2, 8 cols.
3. **Sustained TOPS (measured): 5.67 avg / 5.80 best at 17408×5120×512
   (gate/up); 4.81 / 4.94 at 5120×17408×512 (down); 4.28 / 4.64 at 2048³.**
   All three PASS vs numpy. Best hipfire-shape TOPS **5.67**.
4. All three shapes exact-legal (no padding); tiles 64/64/32, npu2 8-col, Peano.
5. Contention %: G0/C not started — waiting for HaloLF7 to release gfx1151.
   Informational only given K1 FAIL. Theoretical split win dies at s≳10%.
6. IU4 gate/up: **NO backend** — 16.1 ms NPU vs 1.53 ms GPU; K1 FAIL.
7. IU4 down: **NO backend** — 19.0 ms NPU vs 1.57 ms GPU; 4.81 TOPS.
8. qkv/o GEMM: **NO** — launch-bound tiny calls + expansion/fold taxes.
9. FA2 / GDN / small kernels: **NO** — softmax, sequential recurrence, BW-bound.
10. **Kill verdict: K1 FAIL (5.67 < 15 TOPS) → close NPU prefill GEMM, no Sol spec.**
