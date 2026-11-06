# task-graph certificate fixtures

Real MegaM0 P0 pool objects (PeaceMaker persistent dispatchers hosting lifted
hipcc bodies as regions) and the standalone objects their regions were lifted
from. They are inputs of `tests/task_graph.rs`; nothing here is hand-written.

- `p0_r6_pool_b32.co`, `p0_r6_pool_b64.co`, `p0_r6_pool_b128.co`, `p0_r6_pool_b256.co`: one
  pool per blockDim, each with a `.note.pm_regions` table (b256 also hosts the native COPY
  template 2). These are the OR-poison pools (watchdog poison = device-scope release
  `global_atomic_or_b32`, no swap), generated at the commit that tags regions in
  `lifted_dispatch_gfx1201::pool` (`pmprereq/lift` e7e8af36d5 and later).
- `standalone/<sha256>.co`: the hipcc objects, named by the SHA-256 the note records
  (`certify_pool`'s resolver looks them up by it; the digest is checked).

## Generator

From the repository root (CPU only), with MegaM0's rows-6 template table and its
standalone objects (`release-0.4.5/megam0/p0h/out`):

```
cargo run -p peacemaker-isa --features lab --example pm_lift_pools -- \
    TEMPLATES.json STANDALONE_DIR OUT_DIR
# TEMPLATES.json  = megam0/p0h/out/templates_r6.json
# STANDALONE_DIR  = megam0/p0h/out/standalone   (one <name>/sa.co per object)
# OUT_DIR/p0_r6_pool_b<B>.co are the pools
for f in STANDALONE_DIR/*/sa.co; do cp "$f" "standalone/$(sha256sum "$f" | cut -d' ' -f1).co"; done
```

`lift_common::build_pools` tags every region (`lifted_dispatch_gfx1201::pool`
sets the template id and the lift provenance) and links with
`native::assemble_with_regions`, so the objects carry the region note.

Inspect an object with `railgun-cert pool-dump POOL.co`; certify one with
`railgun-cert pool --standalone standalone POOL.co`.
