# Halo XDNA2 NPU probe — measured results (hipx)

Date: 2026-09-16. Probe agent: HaloNpuProbe. Recipe:
`docs/plans/2026-09-16-halo-npu-screen.md`. Work area on hipx:
`/home/kaden/npu-screen` (user-space only; no hipfire source touched, nothing
committed). sudo was used for exactly the memlock fix (parent-authorized);
all probe runs are user-space.

## Ten-line summary

1. Stack present: `amdxdna` 7.0.0-31, `/dev/accel/accel0` (`root:render`),
   XRT 2.21.75, **RyzenAI-npu5** `[0000:c0:00.1]`, FW **1.1.2.65**.
2. Only blocker was `RLIMIT_MEMLOCK=8 MiB` (XRT 64 MiB `MAP_LOCKED` mmap
   `err=-11`); fixed via `limits.d` + `user@1000` `LimitMEMLOCK=infinity`
   (§1.1). Probe shells: `sudo -n -u kaden -i`.
3. **Sustained TOPS (stock mlir-aie `whole_array` i8→i32, Peano, npu2 8-col,
   tiles 64/64/32, −w10 −i20): 5.67 avg / 5.80 best at 17408×5120×512;
   4.81 / 4.94 at 5120×17408×512; 4.28 / 4.64 at 2048³.** All PASS vs numpy.
4. Effective DRAM: **6.3–7.9 GB/s** (compute-bound at this TOPS, far below
   the paper’s ~50 GB/s GEMM figure). Power mode Default.
5. **Contention: GPU IU4 median slowdown  −0.06% (gate set) / +2.03% (down
   add)** under a hot NPU hammer. K2 (≤20%) would PASS. NPU itself +~2%
   under GPU load.
6. IU4 gate/up: **NO backend** — 16.1 ms NPU vs 2.15 ms GPU (this oracle)
   / 1.53 ms (campaign). K1 FAIL.
7. IU4 down: **NO backend** — 19.0 ms NPU vs 2.15 ms GPU; 4.81 TOPS.
8. qkv/o GEMM: **NO** — 38 µs ×384, launch-bound + int8-expand/fold taxes.
9. FA2 / GDN / small kernels: **NO** — softmax, sequential recurrence, BW-bound.
10. **Kill: K1 FAIL (5.67 < 15 TOPS) → close NPU prefill GEMM, no Sol spec.**
    K2 would pass on this stock pair (~2%); K3 fold-numeric fails by inspection.

---

## 1. What exists on hipx (assignment §1)

```
$ ls -l /dev/accel/
crw-rw---- 1 root render 261, 0 Sep  9 02:14 accel0

$ lsmod | grep -i xdna
amdxdna               172032  0
amd_pmf               131072  1 amdxdna
gpu_sched              69632  2 amdxdna,amdgpu

$ which xrt-smi
/usr/bin/xrt-smi
```

- `uname -r`: `7.0.0-31-generic`; Ubuntu 26.04.1 LTS; Ryzen AI MAX+ 395
  w/ Radeon 8060S (MS-S1 MAX, BIOS 1.06); 32 cores; 31211 MB RAM.
- XRT **2.21.75** via Debian packages (`libxrt2`, `libxrt-npu2`,
  `libxrt-dev`, `libxrt-utils`, `libxrt-utils-npu`, `python3-xrt` —
  all `1:2.21.75+dfsg-4`). No `/opt/xilinx/xrt` (distro layout).
- `xrt-smi examine` (after memlock fix):

```
NPU Firmware Version : 1.1.2.65
amdxdna Version      : 7.0.0-31-generic
|BDF             |Name          |
|[0000:c0:00.1]  |RyzenAI-npu5  |
Platform Name        : RyzenAI-npu5
Power Mode           : Default
Total Columns        : 8
```

- Firmware dir `/lib/firmware/amdnpu/`: `1502_00`, `17f0_10`, `17f0_11`.
- `pyxrt.device(0)` → `RyzenAI-npu5` (only after the memlock fix).
- User is in group `render`. `gfx1151` iGPU is HIP 7.15, `ROCR_VISIBLE_DEVICES=1`.

### 1.1 Blocker found + fix (memlock)

Initial state: `xrt-smi examine` and `pyxrt.device(0)` both failed with

```
mmap(addr=…, len=67108864, prot=3, flags=8209 /* SHARED|FIXED|LOCKED */,
     offset=4294967296) failed (err=-11): Resource temporarily unavailable
```

Root cause: `RLIMIT_MEMLOCK` was **8388608 bytes (8 MiB) soft+hard**;
XRT needs a 64 MiB locked mapping. `/etc/security/limits.d/99-amdxdna.conf`
was absent. Unprivileged raise is impossible (`ulimit -l unlimited` and
`prlimit --memlock=unlimited` → `Operation not permitted`).

Fix applied with parent-authorized `sudo -n` (exact commands, nothing else
with sudo):

```bash
echo "* soft memlock unlimited
* hard memlock unlimited" | sudo -n tee /etc/security/limits.d/99-amdxdna.conf
sudo -n mkdir -p /etc/systemd/system/user@1000.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' \
  | sudo -n tee /etc/systemd/system/user@1000.service.d/memlock.conf
sudo -n systemctl daemon-reload
```

Caveat: already-running login sessions keep the old 8 MiB limit (the
`user@1000` manager predates the drop-in). Working path used for every
NPU run: `sudo -n -u kaden -i …`, where `ulimit -Sl/-Hl` both report
`unlimited`. A reboot or `systemctl restart user@1000.service` would
make plain `ssh hipx` inherit it too; not done here.

## 2. Toolchain (assignment §2, user-space)

- `git clone --depth 1 https://github.com/Xilinx/mlir-aie.git` →
  `/home/kaden/npu-screen/mlir-aie`, main @
  `f50bef713297b7c5287a3ebbce964944354be11f` (2026-09-14).
- `utils/env_install.sh` hard-requires `python3.12`; hipx has only
  python3.14, apt is out of scope. Manual equivalent:
  `python3 -m venv ironenv`,
  `pip install -U -r utils/peano-requirements.txt`
  (`llvm-aie==22.0.0.2026090701+3e93bf7b`), plus the matching wheel
  `mlir_aie==1.4.4.dev3+gf50bef7` from `latest-wheels-4`
  (`utils/find_mlir_aie_wheel.py <HEAD>`). `import aie` OK (cp314).
- Design: `programming_examples/basic/matrix_multiplication/whole_array/whole_array.py`,
  `--dtype_in i8 --dtype_out i32 --dev npu2 --n-aie-cols 8`
  (full 4×8 array), default tiles `m=64, k=64, n=32`, `-w 10 -i 20`.
  Peano, not chess. No extra tiling/DMA work — this is the stock example
  the screen asked for.
- Legality (`M % (m·4) == 0`, `K % k == 0`, `N % (n·8) == 0`) holds
  **exactly** for 2048³, 17408×5120×512, and 5120×17408×512 — no padding;
  payload TOPS == measured TOPS.
- Raw logs on hipx: `/home/kaden/npu-screen/logs/{sanity_2048,gateup_17408x5120x512,down_5120x17408x512,gpu_g0,gpu_c,npu_hammer}.log`.
  Hammer: `/home/kaden/npu-screen/npu_hammer.py` (XRT loop, no numpy verify).

## 3. Solo NPU matmul (assignment §2)

FLOPs: `2·M·K·N`. TOPS = FLOPs / t(s) / 1e12.
GB/s = `(1·M·K + 1·K·N + 4·M·N) / t_npu` (int8 A, int8 B, int32 C, one pass).

Timer: example’s own `run_iters` (host chrono around XRT `kernel.wait()` as
“NPU time”, plus end-to-end). Reports avg/min/max over 20 timed iters after
10 warmup — not a median; spread is tight (gate max/min = 1.07×).

| shape (M×K×N) | role | GFOP | NPU avg µs | NPU min / max µs | TOPS (avg/min) | e2e avg µs | GB/s @avg | PASS |
|---|---|---|---|---|---|---|---|---|
| 2048³ | sanity vs published ~6.7 TOPS | 17.18 | 4018.3 | 3704.0 / 4307.4 | **4.28 / 4.64** | 4828.5 | 6.26 | PASS |
| 17408×5120×512 | IU4 gate/up | 91.27 | **16099.4** | 15735.6 / 16819.9 | **5.67 / 5.80** | 16976.6 | 7.91 | PASS |
| 5120×17408×512 | IU4 down | 91.27 | **18956.9** | 18476.9 / 19614.8 | **4.81 / 4.94** | 19802.0 | 5.72 | PASS |

Reference: community stock whole_array i8 ≈ 6.65 TOPS (Strix Point);
this Halo run is in the same band (4.3–5.7). Paper hero (chess/AIE-API, ~4K)
up to ≈ 38 TOPS. Kill floor K1 = 15 TOPS.

2048³ is **below** the published ~6.7 TOPS (4.28 avg) — same example, different
silicon/stack/power (Default, not turbo); not a regression we chased.

## 4. Concurrency (assignment §3)

GPU command (prebuilt, unmodified; HaloLF7 confirmed not using gfx1151):

```bash
HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
  HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-lf3 HIPFIRE_GFX11_MQ4V2_IU4=1 TIME=1 \
  /home/kaden/wt-lloyd/target/release/examples/tmp_halo_iu4_oracle \
  /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt
```

`TIME=1` prints GPU-event median µs over 100 interleaved samples (10 warm)
for production `occ3_col` (`ref_med`) and `lf16` (`lf_med`) on
`gate_proj/set` and `down_proj/add`. Both G0 and C ended `LF PASS`.

NPU leg: `/home/kaden/npu-screen/npu_hammer.py` looping the gate shape
(compile once, no numpy verify). Hot at ~15.9 ms/iter solo; ~16.4 ms/iter
while C ran.

| arm | G0 ref_med µs (TOPS) | C ref_med µs (TOPS) | slowdown |
|---|---|---|---|
| gate_proj/set occ3_col | 2146.057 (42.53) | 2144.756 (42.55) | **−0.06%** |
| down_proj/add occ3_col | 2152.891 (42.39) | 2196.653 (41.55) | **+2.03%** |
| gate_proj/set lf16 | 1913.422 (47.70) | 1945.903 (46.90) | +1.70% |
| down_proj/add lf16 | 2291.189 (39.83) | 2341.383 (38.98) | +2.19% |

**Headline K2 metric: GPU GEMM median slowdown ≈ 0–2%** on the production
`occ3_col` arms. Well under the 20% kill. NPU TOPS under GPU load ≈ 5.55
(vs 5.67 solo, +2% NPU). Interpretation: stock whole_array only pulls
~8 GB/s, so it barely collides with IU4 on shared LPDDR. A 15–38 TOPS
NPU kernel would move much more data and could look worse; this number is
**stock-example supplemental**, not a hero-kernel contention proof.

**Pinned GPU reference (per Main, for every contention window):**
`/home/kaden/wt-lloyd` @ `7ec3829a82714712afd4095857096e607e244315`
(“wip(halo LF.7): lf16 at 96 VGPR / 0 spill …”, detached HEAD, tree clean;
never checked out or rebuilt — read-only `git log` only). Binary
`/home/kaden/wt-lloyd/target/release/examples/tmp_halo_iu4_oracle`
sha256 `da87a0bce21f5663838e5a02e433aff5bf7894b978f20943e2ff2c1afd4970db`.
JIT kernels (all four, both legs):
`gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151`,
`…_full_add_occ3_col_gfx1151`, `…_full_set_lf16_col_gfx1151`,
`…_full_add_lf16_col_gfx1151`.
Cache note: this probe ran with `HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-lf3` per
the original orders; Main’s pinned cache for later windows is
`/tmp/kc-halo-lf4` with alone-baseline gate ref 2140 / lf16 1856, down ref
2153 / lf16 2248 (this probe’s G0: 2146.057 / 1913.422 and 2152.891 /
2291.189 — same binary, lf3 cache, within run noise except lf16 gate
+3.1%).

## 5. Verdict table (per prefill kernel class, chunk-512 layer slice)

GPU campaign baselines (pp2048 profile, not remeasured except the oracle
in §4): IU4 gate/up 1533 µs @ 91.27 GFOP (≈59.5 dense-equiv TOPS); IU4 down
1566 µs (≈58.3 TOPS); qkv/o 38 µs ×384; FA2 H24/KV4/D256 1.1 ms @ L=2048
(17 ms @ 32K); GDN q8 48 heads HD128 T=512 991 µs; small kernels 40–250 µs.

This oracle’s own GPU medians are slower (2.15 ms gate / 2.15 ms down,
42 TOPS) — still 7–9× faster than NPU at the same shape.

| kernel class | FLOPs/call | bytes/call (NPU i8 path) | NPU feasibility | proj. NPU µs @MEASURED 5.67 TOPS (hero 38) | row-slice offload? |
|---|---|---|---|---|---|
| IU4 gate/up | 91.27 GFOP | A 89.13 MB i8 + B 2.62 MB i8 + C 35.65 MB i32 = 127.4 MB. MQ4 W ≈ 44.6 MB nibbles → **2× int8 expand**. `block_i4_128` X also expands. | Native int8×int8→int32 MAC; **not** gfx11 IU4 WMMA. IU4_FOLD_RN f32 fold needs an explicit AIE/host epilogue (K3). Sequential K accumulate is fine (output-stationary). | **16 099 measured** (2 402 hero) vs GPU 1533 campaign / 2146 this oracle | **NO backend.** See §5.1: theoretical ≤9% IU4 wall at s=0, gone by s≈10%; expand/fold/sync untaxed. K1 FAIL. |
| IU4 down | 91.27 GFOP | A 89.13 + B 8.91 + C 10.49 = 108.5 MB; same expand | same | **18 957 measured** (2 402 hero) vs GPU 1566 / 2153 | **NO** — 4.81 TOPS, worse than gate |
| qkv/o GEMM (`gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds`) | ~1.6 GFOP/call at 38 µs × 42.5 TOPS ([INFERENCE] from GPU time) ×384 calls | small, weight-resident | same expand/fold per call; 384 tiny launches, XRT submit/sync ≫ 38 µs | ≫38 µs (launch-bound) | **NO** |
| FA2 QKᵀ/PV | ~2·512·L·256 per matmul × heads + softmax (L=2048 → ~0.54 GFOP QK + 0.54 GFOP PV per 24-head view) | KV streams L keys | softmax (exp/sum, online renormalize) + causal mask + GQA: no stock NPU path | n/a | **NO** |
| GDN q8 recurrence | small GEMMs + scan, 48 heads HD128 T=512, 991 µs GPU | state carry | **sequential** T=512 recurrence; AIE array starves; no row-split | n/a | **NO** |
| small kernels (rmsnorm/rotate/quant/conv1d/gated_norm/deinterleave) | tiny | streaming, 40–250 µs GPU, BW-bound | NPU launch+XRT round-trip > kernel | n/a | **NO** |

### 5.1 Row-split arithmetic

A slower NPU can still cut wall if it runs **in parallel** on a row
fraction `f`: `wall = max(f·t_N, (1-f)·t_G·(1+s)) + t_sync`.
Using measured gate `t_N=16099 µs` and campaign `t_G=1533 µs`:

| s (GPU slowdown) | optimal f | wall µs | IU4 Δ vs 1533 |
|---|---:|---:|---|
| 0 | 0.087 | 1400 | **−8.7%** |
| 5% | 0.091 | 1464 | −4.5% |
| 10% | 0.095 | 1528 | −0.3% |
| 20% (K2 kill) | 0.102 | 1652 | **+7.8% (loss)** |

Using this oracle’s `t_G=2146 µs` instead: s=0 optimal f=0.118, wall=1898 µs
(−11.6%). Measured s≈2% would leave a similar ~10% untaxed IU4 win.

That is **not** a reopen. The 9–12% is before int8-expand BW, host/AIE fold,
and XRT↔HIP host-wait sync — any one of those eats it. K1 (15 TOPS floor)
still fails by 2.6×.

## 6. Kill verdict (screen §5)

| # | Metric | Result | |
|---|---|---|---|
| K1 | solo NPU int8 TOPS ≥ 15 | **FAIL** — best hipfire shape **5.67 TOPS** (5.80 min-time) | 2.6× below floor; stock example matches the 6–8 TOPS band, not a 15–30 TOPS backend |
| K2 | GPU GEMM slowdown ≤ 20% | **would PASS** at **0–2%** on this stock pair | informational / supplemental; a 15–38 TOPS NPU kernel was not in the machine |
| K3 | row numeric = IU4_FOLD_RN | **FAIL by inspection** | int8→int32 GEMM ≠ MQ4 nibble WMMA + fp16-header f32 fold |

**Disposition: close NPU prefill GEMM.** Do not write a Sol spec. Do not
spend further hipfire time on XDNA2 for pp2048 IU4. A later IRON hero
kernel that actually holds ≥15 TOPS at these shapes could reopen K1;
until then the idea stays in the deferred/rejected bucket
(`docs/plans/2026-09-15-halo-gap-plan.md`).

## 7. Log template (screen §3.3)

```
date: 2026-09-16
host: hipx (Ryzen AI MAX+ 395, Ubuntu 26.04, kernel 7.0.0-31-generic)
xrt-smi: RyzenAI-npu5 [0000:c0:00.1] FW 1.1.2.65  power=Default  cols=8
mlir-aie: f50bef713297b7c5287a3ebbce964944354be11f
peano: llvm-aie==22.0.0.2026090701+3e93bf7b
wheel: mlir_aie==1.4.4.dev3+gf50bef7 (cp314)
tiles: m=64 k=64 n=32  n_aie_cols=8  dtype i8→i32  compiler=Peano

shape 2048 2048 2048  solo: avg=4018.3 us min=3704.0 TOPS=4.28/4.64  GB/s=6.26  PASS
shape 17408 5120 512  solo: avg=16099.4 us min=15735.6 TOPS=5.67/5.80  GB/s=7.91  PASS
shape 5120 17408 512  solo: avg=18956.9 us min=18476.9 TOPS=4.81/4.94  GB/s=5.72  PASS

gpu G0  gate set ref_med=2146.057 us (42.53 TOPS)  down add ref_med=2152.891 us (42.39 TOPS)
concurrent G|N  gate set ref_med=2144.756 us (−0.06%)  down add ref_med=2196.653 us (+2.03%)
N under load  ~16400 us/iter (~5.55 TOPS, +2% vs solo)

kill_tops: FAIL
kill_dram: PASS (stock pair only)
kill_numeric: FAIL (inspection)
```
