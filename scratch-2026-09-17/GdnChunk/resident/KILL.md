# GDN resident C slice — KILL

- Commit base: `61227fef2`
- GPU ordinal: `0`
- K2 + resident median: `1595.668 us` / 512 rows
- Matched serial median: `450.003 us` / 512 rows
- Saving: `-1145.666 us`
- Required saving: `>=68.257 us`
- Resident resources: `170 VGPR`, `21,504 B LDS`, `0 B private scratch`, `0 spills`, `8 waves/SIMD`, `3 active blocks/MP`
- CPU f64 output NRMSE: `NaN`
- CPU f64 output max abs delta: `0.0253348631`
- CPU f64 final-state NRMSE: `NaN`
- CPU f64 final-state max abs delta: `415.924874`
- Resource gate: `ALIVE`
- Numeric gate: `KILL`
- Performance gate: `KILL`
- Final gate: `KILL`

The resident candidate and standalone driver are retained beside this file as untracked evidence. No resident kernel, Rust route, workspace, feature flag, KLD, serving, or paired-daemon integration was committed or funded after this gate.
