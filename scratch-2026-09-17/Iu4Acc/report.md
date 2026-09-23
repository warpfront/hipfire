# IU4 single-f32-accumulator P1

## Verdict

**KILL before GPU execution.** The required compile gate failed: the best bounded single-accumulator lowering uses 256 VGPR, 68 B scratch/lane, and 18 VGPR spill operations in both `full_set` and `full_add`, versus shipped 202 VGPR, zero scratch, zero spills. It therefore frees no register budget for a 4x4 retile. Kernel A/B/A, oracle, and WT2 KLD were intentionally not run on a spilling code object.

## Source formula

For each output cell `(r,t)` and 128-K activation block, the toggle evaluates explicit-RN nodes:

```
p_k32 = RN(sc_w[r] * d_x[t])
acc = RN(p_k32 * float(C_k32[r,t]) + acc)   // four K32 partials
z = RN(zp_w[r] * d_x[t])
acc = RN(z * float(s_x[t]) + acc)           // once per K128
```

Every integer WMMA starts from zero; no integer partial survives a K32 call. The zero term remains once per K128. A literal once/K128 matrix of the 64 lane-local `RN(sc*d)` values exactly replaces the removed 64-dword `cacc`: it compiled at 256 VGPR with 1024 B scratch / 257 VGPR spill ops (`full_set`) and 1032 B / 259 (`full_add`). The bounded variant in the branch therefore rematerializes the identical rounded coefficient at each K32 use and uses a zero-valued dependency chain to keep partials short-lived. Even this best variant spills and is the reported gate result.

## Resource receipts

Commands used JIT-equivalent code generation:

```
/opt/rocm/core/bin/hipcc --genco --offload-arch=gfx1201 -O3 \
  --no-offload-compress [-DHIPFIRE_IU4_SINGLE_ACC=1] \
  -Rpass-analysis=kernel-resource-usage -save-temps ../iu4_combined.hip
```

| kernel | shipped VGPR / scratch / spill ops | bounded single-acc VGPR / scratch / spill ops |
|---|---:|---:|
| generic runtime-add | 202 / 0 B / 0 | 256 / 72 B / 38 |
| `full_add` | 202 / 0 B / 0 | 256 / 68 B / 18 |
| `full_set` | 202 / 0 B / 0 | 256 / 68 B / 18 |

Receipts:

- `baseline/compile.log`: original shipped source.
- `offcheck/compile.log`: final source with toggle off.
- `single/compile.log`: final bounded single-acc source.
- Toggle-off canonical ISA: 10,753 instruction lines in both original and final builds, byte-identical canonical stream, md5 `87a8ec3d296efd4996f4a938e011e5d7`.
- HSACO md5: original baseline `9a0261e5173f3acd4c8820bf15e36761`; final toggle-off `354255a96d7e26e095a3b0af5f54deb7`; single-acc `ef24c9625d49fe3acdf177de57845602`. (ELF hashes include source/debug/container metadata; canonical ISA is the shipped-path identity check.)

## Steady non-terminal K128 ISA census

Same static range method as IU4ATT: one steady non-terminal K128 step of `full_add`; shipped assembly lines 6401..6968 and candidate lines 8469..10010. VOPD halves are counted as separate logical VALU operations.

| quantity / K128 | shipped | bounded single-acc | delta |
|---|---:|---:|---:|
| encoded packets | 557 | 1,339 | +782 |
| WMMA | 32 | 32 | 0 |
| fold/tail logical VALU | 388 | 998 | +610 (+157.2%) |
| fold/tail logical VALU / WMMA | 12.1250 | 31.1875 | +19.0625 |
| scratch load/store instructions | 0 | 12 | +12 |

Candidate fold/tail mnemonic counts (feed/address classes excluded exactly as in the shipped census): `v_cvt_f32_i32` 260, `v_dual_fmac_f32` 111, `v_dual_mul_f32` 129, `v_fma_f32` 56, `v_fmac_f32` 153, `v_mul_f32` 191, dependency `v_xor_b32` 64, and spill/copy `v_dual_mov_b32` 20 + `v_mov_b32` 14 = 998 logical operations.

## GPU gates

- Oracle max |delta| vs shipped: **not measured; compile spill gate killed P1**.
- Kernel A/B/A, gate-up and residual: **not measured; compile spill gate killed P1**.
- WT2 KLD c1/c2/c24 and output md5s: **not measured; compile spill gate killed P1**.

The compile/census result is already stronger than the planned performance threshold: instead of freeing about 64 dwords, the variant reaches the architectural 256-VGPR ceiling, spills, and increases fold/tail logical VALU by 157%. The freed-register premise for P2 is false under unchanged synchronization.
