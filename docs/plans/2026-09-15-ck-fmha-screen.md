# CK tile FMHA forward screen vs hipfire FA2 (gfx1151)

Date: 2026-09-15. Author: CkFmhaScreen (research only).  
Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd` (`mq4-lloyd`).  
**No GPU program was run by this agent.** Parent executes the recipe on **hipx** (`ROCR_VISIBLE_DEVICES=1`, confirm exact `gfx1151`).

## 0. Question and kill threshold

Is Composable Kernel's `ck_tile` FMHA **forward** competitive with hipfire's
`attention_q8_0_fa2_gqa_gfx11` on Strix Halo **gfx1151**?

| Item | Value |
|---|---|
| Incumbent symbol | `attention_q8_0_fa2_gqa_gfx11` |
| Incumbent reference | **7.0 ms/call** at pp8192 mean context (~4352), batch **512** query rows, **48** Q heads / **8** KV heads, **hd=128**, causal (`docs/plans/2026-09-15-halo-gap-levers-max.md:26-27`) |
| Screen dtype | CK pure **fp16** Q/K/V/O (example default) |
| Screen shapes | `batch=1`, `seqlen_q=512`, `seqlen_k ∈ {8192, 32768}`, `nhead=48`, `nhead_k=8`, `hdim=128`, causal |
| **Kill threshold** | CK median ms/call must be **≤ 7.0 / 1.5 = 4.667 ms** at the **8192** shape (i.e. **≥ 1.5× faster** than 7.0 ms). If slower or within noise of 7.0 ms, **do not** open an integration track. |
| Secondary | Also record 32768; no separate kill, but if 8192 passes and 32768 is worse than linear in `s_k`, note bandwidth wall. |

**Apples caveat (must stay in the report):** hipfire FA2 dequantizes **Q8_0 K/V** (and **fwht3-K** on some layers) from packed cache into f16 LDS and takes **f32 Q**; CK example is dense fp16. A CK win on this screen is a *compute-kernel ceiling*, not a drop-in product win. Integration must still pay dequant (or a custom K/V loader) — see §5.

## 1. Support verdict (source-verified on CK `develop`)

Investigated: `https://github.com/ROCm/composable_kernel` shallow clone  
`HEAD 8996ea442bc47a087182876839dfc67146128587` (labeled **Composable Kernel 1.3.0 for ROCm 10.1.0** in `CHANGELOG.md`).

### 1.1 gfx1151 / gfx11 wave32 FMHA fwd: **YES**

| Fact | Source |
|---|---|
| FMHA example filters targets to `gfx9\|gfx1[12]` | `example/ck_tile/01_fmha/CMakeLists.txt:5-6` |
| Codegen factory for `gfx115*` is **separate** from plain gfx11 | `get_factory`: `target.startswith("gfx115")` → `KernelComponentFactoryGfx115` before `gfx11` (`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` ~1568–1581) |
| Plain gfx11 arch trait **excludes** gfx115: `defined(__gfx11__) && !defined(__gfx115__)` | same file, `KernelComponentFactoryGfx11` |
| gfx115 factory subclasses gfx11 tiles; retunes (64,64) and (256,256) only | `KernelComponentFactoryGfx115` |
| **hd=128 tile is identical** on gfx11 and gfx115 | `FmhaFwdTileSize(128, 64, 32, 128, 32, 128, 8,1,1, 8,1,1, 16,16,16, 16,16,16, 6)` i.e. block M0=128, N0=64, K0=32, N1=128, K1=32, warp 16×16×16 WMMA, occupancy hint 6 |
| Pipelines on gfx11/115: **`qr`** and **`qr_hpad`** only (row V). **No** `qr_async` / `qr_async_trload` on ≥gfx11 (TODO still in gfx9 rules) | `get_pipelines` for Gfx11; TODO at ~967 |
| Causal + GQA supported in example | README: `-mask=t\|b`, `-h` / `-h_k` with `h % h_k == 0` |
| CMake knows exact `gfx1151` id `0x1151` | root `CMakeLists.txt` `_ck_gpu_target_string_to_id` |
| Head-grouping LLC default for `gfx1151` = 32 MiB | `example/ck_tile/01_fmha/fmha_fwd_head_grouping.hpp` |
| CHANGELOG: "Added gfx11 support for FMHA"; "Improved FMHA forward performance for head dimension 128 on gfx11 and gfx12 by retuning tile selection"; Wave32 via removing fixed `BlockSize` in `make_kernel` | `CHANGELOG.md` (CK 1.2.0 / ROCm 7.13 and 1.3.0 / 10.1.0 sections) |

**Minimal instance set for our causal fp16 d128 batch (receipt 0, optdim 128, filter below) includes** both masked and unmasked `qr` + `qr_hpad` blobs under suffix `_gfx115` when `--targets gfx1151` (verified by running `generate.py` list_blobs locally on the clone).

### 1.2 Tile / pipeline config for hd=128 causal GQA (gfx1151)

For fp16 `(hdim,hdim_v)=(128,128)` on gfx115:

```text
tile:  bm0=128, bn0=64, bk0=32, bn1=128, bk1=32, bk0max=128
       rm0=8, rn0=1, rk0=1, rm1=8, rn1=1, rk1=1
       wm*=16×16×16 (wave32 WMMA)
       occupancy hint = 6
pipelines (default receipt):
  qr      + pad flags spad/skpad = t/t, dpad/dvpad = f/f, vlayout=row
  qr_hpad + pad flags t/t/t/t
traits of interest for product-like run:
  nlogits, nbias, mask (causal), nlse, ndropout, nskip, nqscale, ntrload, nsink
```

GQA is a **host stride / nhead_q vs nhead_k** concern, not a separate tile family: set `-h=24 -h_k=4`.

**Mask choice for fair causal chunk vs long K cache:**  
When `seqlen_q=512` and `seqlen_k=8192`, **top-left** causal (`-mask=t`) makes query `i` see only keys `0..i` (far too little work vs hipfire positions near the end of context). Use **bottom-right** causal (`-mask=b` or `-mask=2`) so local query `i` attends through key index roughly `s_k - s_q + i` (last queries see the full prefix). That matches "512 new rows at the end of an 8192/32768 cache" better than top-left. Record both if curious; **kill decision uses `-mask=b`**.

Optional closer match to mid-prefill mean ctx ~4352: also time `-s_k=4352 -mask=b` (not a kill gate).

## 2. ROCm / hipx assumptions

| Host | Observation |
|---|---|
| This research node | `/opt/rocm/core-10.0/.info/version` → **10.0.0**; `hipcc --version` → **HIP 7.15.26333**, AMD clang 23 |
| Prior hipfire CK sidecar build note | Validated on **ROCm 7.14** / HIP 7.14 (`experiments/flash-attn-ck-sidecar/README.md`) |
| CK `develop` HEAD changelog tip | **CK 1.3.0 for ROCm 10.1.0** |
| hipx | **Unknown here.** Parent MUST `cat /opt/rocm/.info/version /opt/rocm/core*/.info/version` and `hipcc --version` before trusting the build. |

**Assumption for the recipe below:** hipx has a ROCm **≥ 7.13** (when gfx11 FMHA landed in CK notes) and ideally **10.0.x / 10.1.x** matching this CK tip, with `hipcc` able to `--offload-arch=gfx1151`. If hipx is stuck on older 7.x:

- Prefer a CK commit from the **ROCm 7.13 / 7.2** changelog era that already contains "Added gfx11 support for FMHA", **or**
- Use the pinned FA sidecar recipe (`REQUIRED_CK_REV=13f6d635…` + `gfx11_ck_recipe.patch`) only as a **fallback**; that path historically generated with `--targets gfx11` even for gfx1151 and is **not** the current upstream gfx115 split.

Do not mix a ROCm 10 CK tree with a ROCm 6 toolchain.

## 3. Build + run recipe (parent on hipx)

Copy-paste script. No hipfire tree edits required. Writes under `/tmp/ck-fmha-screen-$$`.

```bash
#!/usr/bin/env bash
# CK tile FMHA fwd screen vs hipfire FA2 7.0 ms reference — gfx1151 / hipx only.
# NO integration. Research timing only.
set -euo pipefail

# ---- host gates ----
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-1}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="${ROCM_PATH}/bin:${PATH}"

echo "=== ROCm / device (archive these) ==="
cat "${ROCM_PATH}/.info/version" 2>/dev/null || true
cat "${ROCM_PATH}"/core*/.info/version 2>/dev/null || true
hipcc --version || true
rocminfo 2>/dev/null | awk '/Name:/{print; exit}' || true
# Expect a line containing gfx1151 on the selected device. Abort if not.
ARCH_LINE="$(rocminfo 2>/dev/null | tr '[:upper:]' '[:lower:]' | grep -m1 'gfx1151' || true)"
if [[ -z "${ARCH_LINE}" ]]; then
  echo "FATAL: no gfx1151 visible under ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}" >&2
  exit 2
fi
echo "device_ok: ${ARCH_LINE}"

# ---- pins ----
CK_REMOTE="${CK_REMOTE:-https://github.com/ROCm/composable_kernel.git}"
CK_REF="${CK_REF:-develop}"   # override to a tag/sha if hipx ROCm != 10.x
WORKDIR="${WORKDIR:-/tmp/ck-fmha-screen-$$}"
GPU_TARGET="${GPU_TARGET:-gfx1151}"
JOBS="${JOBS:-$(nproc)}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

echo "=== clone CK ${CK_REF} ==="
if [[ ! -d ck/.git ]]; then
  git clone --depth 1 --branch "${CK_REF}" "${CK_REMOTE}" ck \
    || git clone --depth 1 "${CK_REMOTE}" ck && git -C ck checkout "${CK_REF}"
fi
CK_SHA="$(git -C ck rev-parse HEAD)"
echo "CK_SHA=${CK_SHA}"

echo "=== configure (Ninja, single arch ${GPU_TARGET}) ==="
rm -rf build
mkdir build && cd build
# cmake-ck-dev.sh wipes CMakeCache in cwd; pass arch as 2nd positional.
../ck/script/cmake-ck-dev.sh ../ck "${GPU_TARGET}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCK_BUILD_TESTS=OFF \
  2>&1 | tee "${WORKDIR}/cmake.log"

echo "=== build tile_example_fmha_fwd only ==="
# Full CK is huge; build the example target (+ its generated instance objects).
ninja -j"${JOBS}" tile_example_fmha_fwd 2>&1 | tee "${WORKDIR}/ninja.log"
EXE="$(find . -name tile_example_fmha_fwd -type f | head -n1)"
test -x "${EXE}"
echo "EXE=${EXE}"

run_one () {
  local tag="$1"; shift
  echo
  echo "=== RUN ${tag} ==="
  # -v=0 skip CPU ref (timing only). -kname=1 prints selected instance.
  # -iperm=1 -operm=1 => b,h,s,d (CK default). Product layout may differ; this is the kernel ceiling.
  set -x
  "${EXE}" "$@" \
    -prec=fp16 \
    -mode=0 \
    -bias=n \
    -lse=0 \
    -vlayout=r \
    -iperm=1 \
    -operm=1 \
    -v=0 \
    -kname=1 \
    -warmup="${WARMUP}" \
    -repeat="${REPEAT}" \
    -json=1 \
    -jsonfile="${WORKDIR}/fmha_${tag}.json" \
    2>&1 | tee "${WORKDIR}/fmha_${tag}.log"
  set +x
}

# Primary kill shapes (bottom-right causal; see §1.2)
run_one "q512_k8192_br"  -b=1 -h=24 -h_k=4 -s=512 -s_k=8192  -d=256 -mask=b
run_one "q512_k32768_br" -b=1 -h=24 -h_k=4 -s=512 -s_k=32768 -d=256 -mask=b

# Diagnostics (do not use for kill alone)
run_one "q512_k8192_tl"   -b=1 -h=24 -h_k=4 -s=512 -s_k=8192  -d=256 -mask=t
run_one "q512_k4352_br"   -b=1 -h=24 -h_k=4 -s=512 -s_k=4352  -d=256 -mask=b
run_one "q8192_k8192_tl"  -b=1 -h=24 -h_k=4 -s=8192 -s_k=8192 -d=256 -mask=t

echo
echo "=== parse ms (ave_time field printed as ', X.XXX ms,') ==="
python3 - <<'PY' "${WORKDIR}" "${WARMUP}" "${REPEAT}"
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
kill_ms = 7.0 / 1.5
print(f"kill_threshold_ms = {kill_ms:.4f}  (must be <= this at q512_k8192_br)")
print(f"incumbent_ref_ms  = 7.0000")
for p in sorted(root.glob("fmha_*.log")):
    text = p.read_text(errors="replace")
    ms = re.findall(r",\s*([0-9]+\.[0-9]+)\s*ms,", text)
    names = re.findall(r"kernel[^:]*:\s*(\S+)", text, flags=re.I)
    print(f"{p.name}: last_ms={ms[-1] if ms else 'NONE'} all_ms={ms} kname_hits={names[:3]}")
    if p.name.startswith("fmha_q512_k8192_br") and ms:
        v = float(ms[-1])
        ratio = 7.0 / v if v > 0 else float("inf")
        verdict = "PASS_KILL" if v <= kill_ms else "FAIL_KILL"
        print(f"  -> {verdict}: {v:.4f} ms  speedup_vs_7ms={ratio:.3f}x")
print("WORKDIR", root)
PY

echo "DONE. Archive ${WORKDIR} (cmake.log ninja.log fmha_*.log fmha_*.json CK_SHA=${CK_SHA})."
```

### 3.1 How to read the example timer

The runner prints average kernel time after warmup/repeat as:

```text
, <ave_time> ms, <tflops> TFlops, <gb/s> GB/s
```

(`example/ck_tile/01_fmha/fmha_fwd_runner.hpp` ~1892–1899). That **ave_time is the number to compare to 7.0 ms**.

### 3.2 Build-time notes

- `ninja tile_example_fmha_fwd` still compiles many generated trait combinations (receipt default covers a large pad/mask/bias/lse matrix). Expect **long** first compile (tens of minutes on a fat machine). For a thinner build, after configure you can re-run generate with a tight filter and point the example at those objects — not required for the first screen.
- Generator target string must be **`gfx1151` or `gfx115`**, not bare `gfx11`, or the device-name dispatch may miss Strix Halo (`device_name.compare(0, 6, "gfx115")`).
- Official one-liner from README:  
  `../script/cmake-ck-dev.sh .. gfx1151 -G Ninja && ninja tile_example_fmha_fwd`  
  then  
  `./bin/tile_example_fmha_fwd -b=1 -h=24 -h_k=4 -s=512 -s_k=8192 -d=256 -mask=b -v=0 -warmup=10 -repeat=50 -kname=1`

## 4. Published / known gfx11 FMHA throughput numbers

| Source | What it gives | Useful as kill input? |
|---|---|---|
| CK `example/ck_tile/01_fmha/README.md` | CLI shape examples only; **no ms table** | No |
| CK `script/benchmark_fwd.sh` | Sweep script (FA-v2-like `nhead=2048/hdim`, `s` up to 16k); **no checked-in results** | No |
| CK CHANGELOG | Qualitative “improved hdim 128 on gfx11/12” | No |
| Public web search (this screen) | **No** credible published ms/TFLOP table for gfx11/1151 FMHA fwd at 48/8×128 GQA | No |
| hipfire `experiments/flash-attn-ck-sidecar/README.md` | **End-to-end** Qwen3.6-27B pp8192 on **gfx1100**: native 572.5 tok/s vs CK route 797.4 tok/s (+39%); rocprof **CK FMHA ~283.6 ms / full pp8192 pass** (all layers) for **D256 Asym3** path, plus ~74 ms decode/convert | **Not** a substitute for this microbench: wrong GPU (1100), wrong head dim (256), quantized staging, wall is full model |
| hipfire FA2 profile | 7.0 ms/call mean at pp8192 (caller); 16×512 µs = 8.192 ms bucket at pp512 in an earlier ledger | **Baseline only** |

**Conclusion:** there is **no** published gfx1151 number that can skip the hipx run. The sidecar e2e win on gfx1100 D256 is encouraging for “CK math can beat a packed path when dequant is paid,” but it does **not** clear the 1.5× vs 7.0 ms d128 FA2 micro kill on Halo.

## 5. Integration cost sketch (only if kill PASSES)

### 5.1 Format gap

| Tensor | hipfire FA2 today | CK fmha_fwd example |
|---|---|---|
| Q | **f32** dense rows | **fp16** (or bf16) |
| K | **Q8_0** 34 B/head (`f16 scale + 32×i8`) or **fwht3** 100 B/head (`f32 cnorm + 96 B 3-bit`) | **fp16** dense |
| V | **Q8_0** 34 B/head | **fp16** dense |
| O | **f32** | **fp16** (cast/store) |
| Causal | `positions[]` per row | mask enum + seq lens |
| Layout | internal packed cache strides | arbitrary strides; example defaults b,h,s,d |

Kernel header contract: `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:64-72,115-123`.

### 5.2 Two integration paths

**Path A — dequant to fp16 scratch (matches existing sidecar philosophy)**  
1. Pre-launch (or fused prologue): decode Q8/fwht3 K and Q8 V tiles that the chunk needs into caller-owned fp16 workspace; cast Q f32→f16; after CK, f16→f32 O.  
2. Reuse `experiments/flash-attn-ck-sidecar/` patterns: versioned C ABI, capability table, workspace query, fail-closed Rust loader (`crates/rdna-compute/src/flash_attn_ck.rs`). Extend cells from D64/D256 experiment to **D128 GQA causal** exact-gfx1151.  
3. Cost: extra HBM traffic and kernels every chunk. Sidecar rocprof on gfx1100 D256 showed dequant+convert **~74 ms** beside **~284 ms** FMHA for full pp8192 — order-of-magnitude reference only.  
4. fwht3 layers also need Q rotation (already a FA2 prologue) before CK.

**Path B — custom K/V loader inside a CK pipeline**  
1. Fork `BlockFmhaPipelineQRKSVS` (or policy) to load Q8/fwht3 records straight to the f16 register/LDS fragments CK already uses.  
2. High design cost: must preserve CK’s pad/mask/GQA stride machinery and gfx115 WMMA distributions; fights upstream merge; radiowave/JIT must compile CK headers or ship prebuilt instances.  
3. Only justified if Path A dequant eats the 1.5× margin.

### 5.3 Build implications for JIT-hipcc + radiowave

| Approach | Fit |
|---|---|
| **Prebuilt exact-arch `.so` sidecar** (current experiment) | Best fit. radiowave stays ignorant of CK templates; daemon `dlopen`s ABI; hipcc builds sidecar offline with `-std=c++20 -O3 --offload-arch=gfx1151` and CK includes. Matches `build_sidecar.sh` / `package-ck-runtime.sh`. |
| **JIT CK from hipfire kernel cache** | Poor fit. CK FMHA instances are heavy C++ template TUs; compile times dwarf current HIP JIT kernels; need codegen receipt + many TUs or one monster TU. |
| **Static link CK into rdna-compute** | Binary size + ROCm version coupling; still need instance selection. Avoid unless productizing widely. |

**Minimal product gate if microbench PASSES:**  
1. Rebuild sidecar (or CK example objects) for **gfx1151** with d128 causal GQA cell.  
2. Add Q8→f16 (and fwht3) staging measurement on the **same** 512×8192 shape; kill becomes `(t_dequant + t_ck + t_cast) ≤ 7.0/1.5 ms` **or** explicit wall-transfer ≥1.5× on attention symbol time in-model.  
3. Bit-level oracle vs FA2 is **not** expected (f16 chain ≠ f32 accum FA2); define numeric tolerance separately if ever routed.  
4. Keep FA2 as default until in-model pp8192/32768 ABBA clears methodology docs.

**If microbench FAILS:** archive numbers; no Path A/B work; FA2 levers (F/B6) remain the attention track.

## 6. Decision matrix (parent fills after hipx run)

| Result at `q512_k8192_br` | Action |
|---|---|
| ms ≤ 4.667 (**≥1.5×** vs 7.0) | Open Path A costed spike: dequant+CK on same shape; only then consider product route |
| 4.667 < ms < 7.0 | **No integration** under this kill; optional note “CK fp16 ceiling beats FA2 but <1.5× — not enough after dequant” |
| ms ≥ 7.0 | Hard reject CK as FA2 replacement on Halo for this shape |
| Build/dispatch failure on gfx1151 | Treat as reject for upstream-develop recipe; optionally retry pinned sidecar patch path once; do not block B6/F |

## 7. Sources (URLs and paths)

### Upstream CK (clone HEAD `8996ea442bc47a087182876839dfc67146128587`)
- https://github.com/ROCm/composable_kernel  
- https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/01_fmha/README.md  
- https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py  
- https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/01_fmha/generate.py  
- https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/01_fmha/CMakeLists.txt  
- https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/01_fmha/fmha_fwd_head_grouping.hpp  
- https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/01_fmha/script/benchmark_fwd.sh  
- https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md  
- https://github.com/ROCm/composable_kernel/blob/develop/script/cmake-ck-dev.sh  
- Local clone used for list_blobs: `/tmp/librarian-ck` (ephemeral)

### hipfire tree (read-only)
- `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip` — FA2 contract, Q8/fwht3, 7.0 ms context  
- `docs/plans/2026-09-15-halo-gap-levers-max.md` — 7.0 ms/call citation  
- `docs/plans/2026-09-15-halo-gap-plan.md` — hardware owner hipx  
- `experiments/flash-attn-ck-sidecar/README.md` — prior CK sidecar, gfx1100 e2e, D256  
- `experiments/flash-attn-ck-sidecar/build_sidecar.sh` — offline hipcc recipe, gfx1151 case  
- `experiments/flash-attn-ck-sidecar/gfx11_ck_recipe.patch` — historical gfx11 factory patch (pre-upstream gfx115 split)  
- `crates/rdna-compute/src/flash_attn_ck.rs` — optional loader / fail-closed selection  
- `scripts/package-ck-runtime.sh` — exact-arch bundle helper  

### ROCm on research node (not hipx)
- `/opt/rocm/core-10.0/.info/version` → `10.0.0`  
- `hipcc --version` → HIP 7.15.26333  

---

**Bottom line for the parent:** Upstream CK **does** ship gfx1151-capable fp16 FMHA fwd (qr / qr_hpad, hd=128 tile 128×64×32, GQA+causal). There are **no** published Halo ms numbers; run §3 on hipx. Integration is only on the table if **bottom-right causal** `512×8192` fp16 median **≤ 4.667 ms**, and even then Q8/fwht3 dequant (Path A) must still clear a full-chain bar before touching production attention routing.
