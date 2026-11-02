# Consumer24 K1 on gfx1100

This is a standalone PeaceMaker decode-GEMV port of R5 K1 at `e8a161801`.
It adds files only: it does not register a runtime route or alter existing
code objects. Admission and route wiring are separate work.

K4's serving route admits E8P-I4 cold experts and an MQ4 hot set on exact
gfx1100, not the standalone W2 port described below. W2 is refused on every
architecture and its gfx1100 kernels are not registered for serving. The
codec-generic two-bucket route also admits W3 or E8P cold experts with an
MQ4 hot set on gfx1201 (not both cold codecs within one layer), and W3 or
E8P with an MQ4 hot set on gfx1151 (E8P via its own Halo receipts). gfx1100 W3
remains refused. The kernel instruction bytes and ABI of the existing gfx1201
W3/MQ4 objects and the aggregate MQ4 routes are unchanged; the landed images
are the guarded containers of current beta's code-end baseline
(`fe19435ef`), so container bytes differ from the earlier qualified ones.
The final landing is justified by an exact per-kernel/container proof, not a
new GPU qualification, and post-land gates bind the exact guarded objects.
gfx1100 uses its own K1/K2/K3 and tile-mask images, with AR
serving and MTP off. gfx1151 keeps its beta K1/K2 and hipcc tile mask; its
slab kernel preload is unconditional (even for an MQ4-only layer), the one
exception to the no-cold-kernel-preload rule of gfx1100 and gfx1201.

The gfx1100 MQ4 K1 serving image and module are
`mq_i4_moe_gemv_mq4only_pm_gfx1100`; the mixed W2/MQ4 image is not used.
The E8P K1 module is `e8p_i4_moe_gemv_pm_gfx1100`, and E8P K2 uses
`qwen4_moe_iu4_sym_e8p_pm_gfx1100`. This CPU-prepared wiring is not a GPU
qualification claim: promotion remains gated on the XTX K2 G1.

## ABI and arithmetic

The four entries cover `mq2i4` and `mq4sym`, each with `r256_gate_up`
(`M=1280`, `K=2560`) and `r128_down` (`M=2560`, `K=640`). Symbols append
`_pm_gfx1100`. A workgroup is one wave32; launch grid is `(M,U,1)` and
block is `(32,1,1)`.

The frozen 88-byte little-endian R5 kernarg layout (offsets `0..88`) is unchanged. The
current (rc5) ABI of every K1 image appends a 4-byte `n_experts` and a zero 4-byte
`reserved` word, 96 bytes in all (the R5 receipts and the qualification
text below were taken at the 88-byte ABI):

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u64 | expert_weight_ptrs |
| 8 | u64 | expert_ids |
| 16 | u64 | expert_offsets |
| 24 | u64 | slot_ids |
| 32 | u64 | Xrot |
| 40 | u64 | Y0 |
| 48 | u64 | Y1 |
| 56 | u32 | M |
| 60 | u32 | K |
| 64 | u32 | U |
| 68 | u32 | C |
| 72 | u32 | x_row_div |
| 76 | u32 | x_src_rows |
| 80 | u32 | x_row_stride |
| 84 | u32 | y_slot_stride |
| 88 | u32 | n_experts |
| 92 | u32 | reserved (zero) |

Expert-range guard (all K1 images on every arch: the shared gfx1151/gfx1201 MQ4/W3
images, this MQ4-only image and both E8P images): the workgroup-uniform routing expert
id is compared once, unsigned, against `n_experts` (so a negative id is out of range
too), and an id `>= n_experts` takes the existing no-load, no-store exit before the id
indexes `expert_weight_ptrs`, before any weight or descriptor prefetch and before
the first barrier. The output tile of a skipped workgroup is left untouched (not
zero-filled). `n_experts` is the table extent supplied by the launcher (the
slab launcher passes its `[512]` table's 512). The NT1 K2 images
(`qwen4_moe_iu4_sym*`, gfx1100 MQ4 and E8P, gfx1201 E8P) take the same guard at 64
bytes: the 60-byte ABI plus a trailing `n_experts@60`.

Strides count F32 elements. `expert_weight_ptrs[expert_id]` is the
projection's code-plane base, not an expert-stride calculation. The
address may be VRAM or a device-visible alias of pinned host memory;
residency is not a kernel flag. Slot outputs retain original slot indices.

MQ2-I4 uses planar 2-bit codes, one f16 scale per 128 columns, and
`w = RN(float(d) * float(2*c-3))`. Each lane owns four adjacent elements;
its group dot is a MUL followed by three left-associated FMAs. Group
partials feed four chains for gate/up, one for down. Gate/up folds
`RN(RN(a0+a1)+RN(a2+a3))`. Both then shuffle down 16,8,4,2,1.
MQ4 retains the legacy header decode, element-1-first contraction, group
layout and reduction exactly; it is not reinterpreted as an odd lattice.

## Standalone entry points

All cargo commands must hide GPUs and use a dedicated target directory:

```sh
export ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1
export CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.3/target-consumer24-b2
nice -n 19 cargo run -j8 -p peacemaker-isa --features toolchain \
  --example mq_i4_moe_gemv_gfx1100_emit -- /path/to/emission
nice -n 19 cargo build -j8 -p hipfire-runtime --release --features lab \
  --example mq_i4_moe_gemv_gfx1100_g1
```

The G1 oracle/research examples (`mq_i4_moe_gemv_gfx1100_g1`,
`e8p_i4_moe_gemv_gfx1100_g1`, ...) require the existing `lab` feature; the
native emitters above (`mq_i4_moe_gemv_gfx1100_emit`, ...) keep `--features toolchain`.

The emission example produces `b2.k1.s`, `b2.k1.proof.json`, `b2.k1.co`,
`b2.k1.m7.json` and four per-entry shape contracts. Use `peacemaker custom
build --arch gfx1100 --s ... --out ... --manifest ... --contract ...
--proof ...` for each contract. M7 independently replays gfx1100's
TRANS-use hazards; an ordinary instruction census alone is not that check.

## E8P-I4 (K-E2) entries on gfx1100

The two E8P-I4 entries `e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100` (`M=1280`,
`K=2560`, `x_row_div=10`) and `e8p_i4_moe_gemv_r128_down_pm_gfx1100` (`M=2560`,
`K=640`, `x_row_div=1`) live in the module `e8p_i4_moe_gemv_pm_gfx1100`.
They are the arithmetic of the gfx1201 K-E2 generator, not a rewrite: the
gfx1100 arm is added to `kernels/e8p_i4_moe_gemv.rs` because that is the
smaller proven edit (every difference is behind the architecture switch, so
at the standalone port the gfx1201 text, object and `e8p_i4_moe_gemv_{r256_gate_up,r128_down}`
symbols were unchanged by construction; the rc5 `n_experts` guard is an ABI change of both
archs' E8P images; only the exported gfx1100 names carry
the `_pm_gfx1100` suffix). Entry points are standalone; nothing is added to a
shared registry and no runtime route is wired.

The ABI is identical to the table above: 96-byte kernarg, grid `(M,U,1)`,
block `(32,1,1)`, one wave32 per workgroup, same guards (row `>= M`, expert
index `>= U`, expert id `>= n_experts` (unsigned), slot count `0`, `> 8`, or `offsets[u+1] > 10*C` return without a
store and before any LDS access). `expert_weight_ptrs[e]` is the planar E8P
tensor `[M,K]`: `M*K/8` little-endian `u16` words (`idx | sign7<<8 |
shift<<15`, eight weights each) followed by `M*K/128` f16 scales. Every scale
from the production writer is finite with bits `>= 0x0400`; the raw decode
also handles finite subnormal scales exactly (covered by G1). The kernel
reads exactly `[0, M*K/4 + M*K/64)` and needs no tail padding.

Decode (per lane, four weights of a K128 group; lane `l` owns word `l>>1`,
half `l&1`): `x = LUT[idx] ^ MASK[sign7]`, `y = MASK & 0x11111111`,
`n = shift ? 0xeeeeeeef : 0`, `operand = x + y + n`, `t = operand >> 16*half`,
`v = 2*bfe_i32(t, 4i, 4) + 1`, `w = RN((f32(d)*0.25) * f32(v))`. `LUT[256]`
(offset 0) and `MASK[128]` (offset 1024) are staged once per workgroup into
1,536 bytes of LDS and published at one `s_barrier`. The group dot is a MUL
followed by three FMAs, gate/up uses four chains folded `RN(RN(a0+a1)+RN(a2+a3))`,
down uses one chain, then the shuffle-down at 16,8,4,2,1; lane 0 stores.
The exact CPU reference is `e8p_i4_reference`; G1 is bit-for-bit.

The tables are the unchanged K-E1 E8P codebook: LUT nibbles encode `2|g|`
with lane 7 two's-complement-negated for odd-parity absolute rows; MASK
contains `0xf` in each negated lane, with lane 7 sign equal to the parity
of the seven explicit sign bits. No codebook pointer is added to the ABI.

gfx1100-forced differences from the gfx1201 emission, and nothing else:
workgroup ids are the system SGPRs after the declared user SGPRs (declared in the
register plan, no `ttmp` and no `0xffff` mask): gfx1100's
`FeatureUserSGPRInit16Bug` pads the user SGPRs to user + system >= 16, so x / y
are `s14` / `s15` (the `_t8` entries, which also enable z: `s13` / `s14` /
`s15`) and the kernarg pointer block is `s[88:95]` instead of `s[8:15]`
(gfx1151 keeps `s2` / `s3` / `s4` and `s[8:15]`); the barrier is `s_barrier`;
waits and `s_waitcnt_depctr`/`s_delay_alu` hazard guards are the builder's
gfx11 ones (never hand-placed). The f16 scale widening is
`v_cvt_f32_f16_e64 vD, vD.l` on both architectures, the spelling the
qualified MQ gfx1100 entries already use. The shape contract
(`e8p_i4_moe_gemv.gfx1100.<tag>.contract.json`) therefore counts those and
does not forbid `s_waitcnt*`, while forbidding every gfx12 spelling.

Emission (standalone, GPU-free; `--arch` defaults to `gfx1100`, and
`--arch gfx1201` re-emits the committed gfx1201 module for byte comparison):

```sh
export ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1
export CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.3/target-consumer24-k1e8p
nice -n 19 cargo run -j8 -p peacemaker-isa --features toolchain \
  --example e8p_i4_moe_gemv_gfx1100_emit -- /path/to/emission [--arch gfx1100|gfx1201]
```

It writes `e8p_i4_moe_gemv_pm_<arch>.{s,proof.json,co,hxaco,m7.json}` and the
two `e8p_i4_moe_gemv.<arch>.<tag>.contract.json` files, and prints the object
and bundle SHA-256. The `.hxaco` is `native::bundle` with host entry
`host-x86_64-unknown-linux-gnu`.

The XTX G1 (`e8p_i4_moe_gemv_gfx1100_g1`) passed 37 cases: 22 synthetic
and 15 directly packed, SHA-bound real-weight cases from both E8P artifacts,
including VRAM, mapped-host and split placements. It compared 5,014,656
output words with zero mismatching bytes, checked 712 guarded allocations,
covered all 65,536 wire words at all 16 lane pairs in both projections,
and detected all 71 negative checks (including wrong LUT entry, sign mask,
scale row, expert pointer and swapped lane pair). Synthetic scales include
f16 maximum and subnormals. Artifact extraction is reproducible with
`tools/e8p_k1_extract_samples.py`; only bound sample files and manifests
need copying to the device host.

The standalone `e8p_i4_moe_gemv_oracle` Extreme fixture now selects LUT rows
containing the maximum component, with its CPU gate requiring decoded magnitude
11; the former norm-48 row selection reached only 7. Historical G1 inputs and
results above are unchanged; no new hardware qualification is implied.

## MQ4-only landing object

Only the MQ4 symmetric entries land; the MQ2 (W2) entries do not. A separate
emitter builds `kernels/mq_i4_moe_gemv_mq4only_pm_gfx1100.hxaco` from the
existing qualified generator, unmodified, keeping just
`mq_i4_moe_gemv_mq4sym_{r256_gate_up,r128_down}_pm_gfx1100`:

```sh
export ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1
export CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.3/target-consumer24-k1e8p
nice -n 19 cargo run -j8 -p peacemaker-isa --features toolchain \
  --example mq_i4_moe_gemv_mq4only_gfx1100_emit -- /path/to/emission
```

It writes `mq_i4_moe_gemv_mq4only_pm_gfx1100.{s,proof.json,co,hxaco,m7.json}`
and the two `mq_i4_moe_gemv.gfx1100.mq4sym_*.contract.json` files. Each
function must equal the matching function of the committed four-symbol
`mq_i4_moe_gemv_pm_gfx1100.hxaco` byte for byte; if any differs, the landing
stops. The four-symbol object is re-emitted unchanged by the original
`mq_i4_moe_gemv_gfx1100_emit` example.

The integration run proved both MQ4 function byte strings unchanged and
the legacy four-symbol gfx1100 and two-symbol gfx1201 bundles byte-identical.
The existing G1 with `--mq4-only` passed 25 cases, 57 negative checks and
five old-kernel parity cases with zero mismatches. M7 passed both new
bundles; the generator and new oracle CPU suites passed 21 and 8 tests.
Receipts (outside the checkout):
`/home/kaden/qcal/release-0.4.3/consumer24/rcpt/k1e8p.emit.json` and
`/home/kaden/qcal/release-0.4.3/consumer24/rcpt/k1e8p.g1.json`.

## Gates and measurement

Use the standalone emission and runtime G1 examples, not a shared registry
entry. Shape certification, M7 byte obligations and the hazard audit are
required. G1 is bit-for-bit (including signed zero), not a tolerance test.
Synthetic fixtures precede c24 sample runs, with pinned-host aliases and
negative controls included. A negative control must be detected; it cannot
count as a successful data case.

GPU runs belong to the fabric scheduler on hipx, class `correct` or
`timing`, card `xtx`, owner `Consumer24Lead3.K1E8p1100`. Never use CU masks.
The expected card is `GPU-43390a851e296ee5`, BDF `0000:67:00.0`, `gfx1100`.
Every speed or bandwidth number has this caveat: hipx's XTX is behind a
**PCIe 4.0 x4 root port (`00:03.1`)**. Raw kernel timings are not an
end-to-end decode-throughput claim.

Receipts are stored outside the checkout under
`/home/kaden/qcal/release-0.4.3/consumer24/rcpt/b2.k1.*`.
