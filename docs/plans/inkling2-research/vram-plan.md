# Inkling-2-Mini VRAM plan — text-only AR, max_seq 32768 cap (gfx1100 24GB + Strix Halo note)

Read-only arithmetic note (2026-11-21). No `crates/*` changed. Inputs:
`docs/plans/local/2026-11-20-inkling2-mini-support.md` §0/§6 (24 phys layers,
48 visits, 24Q/8KV hd128, d_rel16, MoE 128× top-8 ×1152, n_embd 2304,
vocab 200058, 4 full + 20 SWA-1024),
`docs/plans/local/inkling2-tensor-mapping.md` shapes, artifact
`D:\hipfire-models\inkling2-mini-mq4.hfq` (13,927,188,480 B, `ls` 2026-11-21;
not rehashed). Qwen35 sizing formulas mirrored with `file:line` cites.

## 0. Geometry used (all from the two input docs)

- Physical 24, visits 48 (core 6–18 ×3 passes). Each visit owns KV + conv state.
- Full-attention physical layers: 5, 11, 17, 23 (4). Core full layers 11, 17 run
  3× → full **visits** = 2 (layers 5, 23) + 2×3 (layers 11, 17) = **8**;
  sliding visits = 48 − 8 = **40** (SWA-1024).
- Per-visit attention: n_kv_heads 8, head_dim 128 → ekv = 1024; eattn = 3072;
  erel = 384. Conv kernel 4. Unified conv width
  2·ekv + 2·n_embd = 2·1024 + 2·2304 = **6656** floats (§5.3/offset table in
  support plan §0a).
- MoE per layer: 128 experts, top-8, inter 1152, no shared experts.

## 1. Weights resident

- File: **13,927,188,480 B = 13.927 GB decimal = 12.971 GiB**.
- Census (tensor-mapping §3): 6144× qt13 MQ4G256 experts + 170× qt44 MQ4G256V2
  (144 attn + 24 routers + embed + unembed) + 476× qt1 F16 keep.
- VRAM rule: quantized tensors stay packed on GPU (like any MQ4 load); F16
  keeps (controllers/queries/norms/biases/clips/convs/rel tables/sinks) land
  verbatim — already inside the 13.93 GB. No separate "F16 overhead" beyond
  alignment/VMM granule rounding + the loader's fixed `slack: 128 MiB`
  (`crates/hipfire-arch-qwen35/src/carrier.rs:375`).
- Budget line: **weights ≈ 12.97 GiB + 0.125 GiB slack ≈ 13.10 GiB** before KV.

## 2. Per-visit KV cache (mirror of the qwen35 formula)

Qwen35 precedent `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:2247-2279`
(`vmm_kv_token_bytes`): per-token bytes = `n_kv_layers × n_kv_heads ×
(k_head + v_head)` with per-vector `k_head/v_head` =
Fp8 `head+2`, Q8 `head/32*34`, F16/BF16 `head*2` (`KvPair::Native` → v=k).
Same table applied here with n_kv_heads 8, head 128:

| kv mode | per vector (B) | per visit·token = 8·(k+v) (B) |
|---|---|---|
| q8 | 128/32·34 = **136** | 8·272 = **2176** |
| fp8 (native) | 128+2 = **130** | 8·260 = **2080** |
| f16/bf16 | 128·2 = **256** | 8·512 = **4096** |

KV slot count at cap 32768, SWA-aware (full visits keep 32768, SWA keep 1024):

- full slots = 8 × 32768 = 262,144; swa slots = 40 × 1024 = 40,960;
  **total = 303,104 visit·slots**.

| kv mode | bytes = 303,104 × row | decimal | GiB |
|---|---|---|---|
| q8 (default plan) | 303,104 × 2176 = **659,554,304** | 0.660 GB | **0.614 GiB** |
| fp8 | 303,104 × 2080 = **630,456,320** | 0.630 GB | 0.587 GiB |
| f16 | 303,104 × 4096 = **1,241,513,984** | 1.242 GB | 1.156 GiB |

Naive (fail-closed overprovision: all 48 visits × 32768) for comparison:
1,572,864 slots × 2176 = **3,422,552,064 B = 3.423 GB (3.188 GiB)** q8.
Size the allocator SWA-aware; quote the naive number only as the refusal
ceiling. SWA mask policy mirrors qwen35 `KvLayers::Mask` (only FullAttention
layers in the stride, `prefill.rs:2271-2276`; VMM owner shaped by mask,
`forward_slots/vmm.rs:496-507`).

## 3. Unified conv state (mirror of the qwen35 conv formula)

Qwen35 precedent `crates/hipfire-arch-qwen35/src/qwen35/weights.rs:4209-4211`:
`conv_channels = 2·(k_heads·k_dim) + v_heads·v_dim`,
`conv_state_size = conv_channels × (kernel − 1)`; same construction in the
lane view (`weights.rs:3910-3912`). Applied here:

- width 6656 × (4−1) = **19,968 floats/visit** → ×4 B = **79,872 B/visit (F32)**.
- ×48 visits = **3,833,856 B ≈ 3.83 MB (0.0036 GiB)** F32; F16 keep halves it
  to ≈ 1.9 MB. Negligible next to KV — but it is **per-visit**, not per-layer
  (×48, not ×24); an implementation that allocates ×24 silently aliases passes.

## 4. MoE scratch (top-8 × 1152)

Grouping precedent `prefill.rs:980` (slots layout `rows × top_k`). Per
visit·token, f32 working set:

- gate+up activations: 8 experts × 2 × 1152 = 18,432 floats;
  down accumulation: 2304 floats → 20,736 floats × 4 B = **82,944 B ≈ 81 KiB
  per visit·token**.
- Decode (1 token): same 81 KiB live per visit, reused across visits — **not**
  ×48 simultaneously unless the forward holds all visits at once (don't).
- Prefill chunk C: ×C → C=1024 ≈ **81 MiB**; C=2048 ≈ 162 MiB; C=4096 ≈
  324 MiB. Chunk the MoE the same way as attention; the chunk row count, not
  max_seq, sets this term.

## 5. Activation high-water: prefill 32768 vs decode

- Decode per step: 1 token × 48 visits through hidden 2304 f32 (9 KiB) +
  eattn 3072 + MoE §4 → order **≈ 5–10 MB** live. Irrelevant to the cap.
- Prefill must be chunked: a full 32768-row QK block on one full visit alone
  is 24 heads × 32768² scores — never materialize; per-chunk live set ≈
  C × (hidden + eattn + MoE §4) × 4 B. At C=1024 ≈ **~110 MB**;
  C=4096 ≈ **~440 MB**. The repo's PBS precedent charges only the minimum
  512-row scratch against the lifetime bound and admits larger PBS lazily
  from post-KV free VRAM (`prefill.rs:2281-2283`; placement spends
  `minimum_prefill_reservation_bytes`, `carrier.rs:353-355`). Copy that:
  **KV is the lifetime reservation; prefill scratch is admitted per request.**
- Minimum lifetime non-KV reservation to carry: prefill floor (few ×10² MB) +
  128 MiB slack. Use **0.75 GiB** as the napkin figure.

## 6. Total vs 24 GB budget (gfx1100 RX 7900 XTX)

24 GB card = 24 GiB class (≈ 25.77×10⁹ B addressable; use 24.0 GiB = 25.77 GB
as the yardstick and keep the 128 MiB slack inside).

| config | weights+slack | KV (SWA-aware) | conv | prefill floor | total | vs 24 GiB |
|---|---|---|---|---|---|---|
| q8, cap 32768 | 13.10 | 0.61 | 0.004 | 0.75 | **≈ 14.46 GiB** | **FIT, ≈ 9.5 GiB headroom** |
| fp8, cap 32768 | 13.10 | 0.59 | 0.004 | 0.75 | ≈ 14.44 GiB | FIT |
| f16 KV, cap 32768 | 13.10 | 1.16 | 0.004 | 0.75 | ≈ 15.01 GiB | FIT |
| q8, naive 48×32768 | 13.10 | 3.19 | 0.004 | 0.75 | ≈ 17.04 GiB | FIT (wasteful — fix sizing, don't ship) |

**Verdict: FIT.** Even the naive overprovision fits; the SWA-aware plan leaves
≈ 40% of the card free for PBS growth and VMM granule rounding.

## 7. What gets spilled first if over (offload precedent)

- Knobs (unchanged): `memory.gpu_layer_budget` (N keeps last N layers resident;
  `auto`/`-1` fits to capacity; precedent: AGENTS.md flag table +
  `carrier.rs:330-381` placement inputs) × `memory.offload_exec`
  (`pcie`/`cpu`/`passback`, `hipfire-config/src/lib.rs:859-881`). **KV stays
  in VRAM either way** (same help text) — spill weights, never KV.
- Order if a future config (longer ctx, bigger chunk, second request) overruns:
  1. prefix trunk layers to host-mapped (`gpu_layer_budget = auto` first;
     manual N only as a pinned diagnostic); 2. downshift prefill chunk / PBS
     before touching max_seq; 3. `offload_exec=pcie` default, `cpu` if the
     link is the bottleneck, `passback` only via `hipfire offload-bench`
     (share `auto`).
- Phase-1 has no draft sidecar, so no draft reservation competes with the
  trunk (contrast qwen35 `draft_reservation`, `carrier.rs:365`).

## 8. Refusal line when max_seq must drop

If `weight_capacity(free, kv(max_seq), prefill_floor, draft=0, slack)` fails
(`carrier.rs:396-402` pattern), refuse fail-closed — never silently truncate:

```text
inkling2: refuse max_seq=32768: needs {kv_bytes} KV ({visits} visits) +
  {weights} weights + {floor} prefill floor > {free} free on {arch};
  lower max_seq (e.g. --max-seq 16384 halves full-visit KV) or set
  memory.gpu_layer_budget=auto
```

Scale rule: KV ∝ full-visit rows, so max_seq 16384 → full slots 131,072,
total 172,032 slots → q8 **≈ 0.374 GB**; max_seq 8192 → **≈ 0.229 GB**.

## 9. Strix Halo note

Same arithmetic; different pool: unified memory, `memory.oom_guard=auto` is ON
for APUs (AGENTS.md flag table) so overshoot is a global OOM, not a HIP OOM.
Budget against the shared pool minus OS/driver reserve, not 24 GB. No PCIe
spill penalty — `offload_exec=cpu` is cheap and `gpu_layer_budget=auto` spills
to shared RAM the GPU still reads fast. Cold-JIT Halo rows can differ in chunk
0 (quality-pin section of AGENTS.md §5) — warm the kernel cache once before
measuring any §5-class number on this model.

## 10. Re-run table (napkin script)

```python
# python3 napkin.py --max-seq 32768 --kv q8
import argparse
p = argparse.ArgumentParser(); p.add_argument("--max-seq", type=int, default=32768)
p.add_argument("--kv", default="q8", choices=["q8", "fp8", "f16"])
p.add_argument("--weights", type=float, default=13927188480)
a = p.parse_args()
FULL, SWA, SWA_WIN = 8, 40, 1024
per = {"q8": 2176, "fp8": 2080, "f16": 4096}[a.kv]
slots = FULL * a.max_seq + SWA * min(a.max_seq, SWA_WIN)
kv = slots * per
conv = 48 * 6656 * 3 * 4
floor = 0.75 * 2**30
total = a.weights + 128 * 2**20 + kv + conv + floor
print(f"slots={slots} kv={kv/1e9:.3f}GB conv={conv/1e6:.2f}MB total={total/2**30:.2f}GiB")
print("FIT" if total < 24 * 2**30 else "REFUSE: drop --max-seq or set memory.gpu_layer_budget=auto")
```
