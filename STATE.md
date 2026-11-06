# STATE: railgun gfx1100 transport port (branch railgun-gfx1100-045, off origin/beta 19ae6fa3cb)

Stopped 2026-11-02 on Main's scope change (gfx1100 railgun out of scope this cycle). Local commit only; no push, no
land-gate, no release. Design note: `docs/design/2026-11-02-railgun-gfx1100.md` (full table of what is ported, G4
protocol, certification list, estimate).

## Done (CPU only, default off, gfx12 behaviour untouched)
- `railgun::plan`: `CacheTable::for_arch("gfx1100")` (RAW/WAR/WAW = WaitIdle + same-agent acquire; pre-writer = bare WaitIdle, probe rung `wait`, receiptable),
  rung/packet doc table, explicit unpaced `ArchLowering` for gfx1100; plan tests (table, tier-D plan, strict plan,
  receipt attach, refusals).
- `rdna-compute::replay`: `Pm4Commands::{system_acquire, architecture, gfx11_dwords}`.
- `railgun_lower.rs`: arch-neutral `emit_segment_with`, `check_gfx11_stream` (no NOP/COND_EXEC/fence/gfx12 GCR),
  `native_dimensions` (gfx11 = workgroup counts), `DeviceSelect` + `device_refusal` (exact gfx1100 by PCI only; gfx1151,
  other gfx11, gfx10 refused), gfx11 branch of `lower_owned`/`patch_grid`/`verify_state`, program-arch identity check.
- `railgun_window.rs`: `railgun_lowering_context` selects gfx1100 by the HIP device's PCI bus address.
- `redline-dispatch`: test that a real gfx11 builder stream decodes/patches in workgroup counts.

## Verification actually run (CPU, via build-slot, `nice -n 19`)
- `cargo test -p railgun --lib plan::` 11 passed (6 existing gfx12 plan tests unedited, 5 new gfx1100 tests).
- `cargo test -p rdna-compute --lib railgun_lower` 8 passed (all new).
- `cargo test -p redline-dispatch --lib gfx11_builder` 1 passed (new).
- NOT run: the full rdna-compute / railgun / redline-dispatch suites (only filtered runs), clippy, fmt, land-gate, any device.

## Left
- Never run on a device: PCI/ROCr match, kernarg pool, IB creation, geometry patch, self-check, replay on the XTX.
- G4 probe on the XTX (>= 2^20 trials/rung) and receipts; until then products must use `RowPolicy::UnreceiptedSystem`.
- `railgun-cert coverage` for the exact gfx1100 Raon objects (peacemaker gfx1100 table lift gaps unknown).
- Owner wiring in `hipfire-arch-raon/src/railgun_decode.rs` (`ARCH = "gfx1201"` on raon-railgun-stack-045).
- Byte oracle + perf gate on the XTX. M1/M2 shadow diff for gfx1100 not ported (stays gfx12-only, refuses).
- Rebase onto beta ecbf8bc274 (MQ8 landed) before any land-gate; full (not filtered) test suites.

## Risks
- The gfx11 pre-writer row (bare WaitIdle) is tier-D only; whether gfx1100 needs a stronger writer-side acquire is a G4 question (live XTX G4 run: RaonG4X, `raon-railgun/g4-gfx1100/`).
- Certification effort is the unknown: 1-5 days. Estimate for the whole remainder: ~5-10 working days.
- `replay.rs` is touched (3 small methods); expect merge friction with other branches editing it.
