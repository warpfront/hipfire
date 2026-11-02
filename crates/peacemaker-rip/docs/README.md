# `.rip` documentation

`.rip` is PeaceMaker's Rust-shaped kernel script language for RDNA (gfx1100,
gfx1151, gfx1201; wave32). A script runs at compile time and emits instructions
into a checked builder; `peacemaker-rip` compiles it to AMDGCN assembly without
rustc.

Start here:

1. [`cookbook/01-minimal-kernel.md`](cookbook/01-minimal-kernel.md): a complete
   small kernel, with each part explained.
2. [`REFERENCE.md`](REFERENCE.md): the language, the builder and Workgroup
   vocabulary, the per-arch launch ABI and instruction spellings, and waits.
3. [`ERRORS.md`](ERRORS.md): every diagnostic code with a failing example and
   its fix. `peacemaker-rip explain CODE` prints the same text.

## The loop

```sh
peacemaker-rip check kernel.rip --variant NAME          # both RDNA arches, every error with a fix-it
peacemaker-rip check kernel.rip --variant NAME --json   # the same, as JSON
peacemaker-rip explain R0301                            # long-form explanation of a code
peacemaker-rip fmt kernel.rip                           # rustfmt layout, tokens verified unchanged
peacemaker-rip build kernel.rip gfx1201 NAME out.s      # write the assembly
```

`check` runs the whole pipeline for gfx1151 and gfx1201: parse, evaluate, the
Workgroup driver's LDS/wait/barrier checks, per-instruction encoding validation
for the target arch, defs/uses validation, native assembly, and the
register-definedness check. It then prints the VGPR, SGPR and LDS use per target.

## Cookbook

Every recipe kernel is compiled and run on the CPU wave32 emulator against a Rust
model on both arches (`tests/cookbook_a.rs`, `tests/cookbook_b.rs`).

| recipe | pattern |
|---|---|
| [01 minimal kernel](cookbook/01-minimal-kernel.md) | spec, plan, `emit`/`author` split, exec-mask tail |
| [02 elementwise bounds](cookbook/02-elementwise-bounds.md) | buffer descriptors as the bounds check |
| [03 workgroup ids](cookbook/03-workgroup-ids.md) | `workgroup_id`, 2-D grids, the per-arch registers |
| [04 wave reduction](cookbook/04-wave-reduction.md) | DPP `row_xmask` + `v_permlanex16_b32`, gfx11/gfx12 spellings |
| [05 LDS handoff](cookbook/05-lds-handoff.md) | regions, `begin_write`/`ds_store`, `wait`, `barrier`, `ds_load` |
| [06 device loop](cookbook/06-device-loop.md) | runtime trip counts: `loop_until`/`break_if`, `loop_carried` |
| [07 uniform branch](cookbook/07-uniform-branch.md) | `skip_if`/`skip_unless`, `wave_exit_unless`, labels |
| [08 WMMA](cookbook/08-wmma.md) | 16x16x16 bf16 fragment layouts on gfx11 and gfx12 |
| [09 kernarg loads and waits](cookbook/09-kernarg-loads-and-waits.md) | kernarg layout, `smem` widths, automatic waits, counters |
| [10 SWMMAC](cookbook/10-swmmac.md) | gfx12 sparse (2:4) WMMA: compressed A, index word, `Wmma::swmmac` |
| [11 production blocks](cookbook/11-production-blocks.md) | `iu4::*` phase blocks, a K-loop composed in script, `Iu4Gemm::emit` |
| [12 mq6 blocks](cookbook/12-mq6-blocks.md) | `mq6::*` instruction builders, LDS rings and split barriers in script: move a wait, a signal, a decode, a fold |
| [13 fp8 blocks](cookbook/13-fp8-blocks.md) | `fp8::*` on the raw `Builder`: barrier lists, loop and branches as script statements, what the Builder refuses |
| [14 qwen4_moe_sym blocks](cookbook/14-qwen4-moe-blocks.md) | `moent::*` / `moe1::*`: per-tile branches, wait placement and the gate/up LDS hand-off as script statements |
| [15 fa_prefill blocks](cookbook/15-fa-prefill-blocks.md) | `fa::*` on the raw `Builder`: two schedules (two planes / one reused stage), subtiles, barrier lists, what the Builder refuses |
| [16 producers](cookbook/16-producers.md) | `c8::*`, `rot::*`, `gdn::*` on the raw `Builder`: reorder the norm's phases, the prefetch and the LDS exchange in a fused producer |
| [17 fine iu4 K-loop natives](cookbook/17-iu4-kloop-natives.md) | `iu4::fragments_body` (the script body `rip_fragments`), `wmma`, `publish_*`, `wait`, `fetch_*`, ...: write the K128 block from typed steps, move the wait/publish placement, what the driver refuses |

Every ```` ```rip ```` block in these documents is compiled by
`tests/docs.rs`, so the examples track the compiler.
