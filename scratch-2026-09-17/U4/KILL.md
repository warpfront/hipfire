# U4 epilogue fast-path: KILL

Candidate source hash: `5b6e9505d0c3a0b2`

Pinned daemon MD5:

- baseline `56462d2e844d33241fa103d7db8dadb7`
- candidate `42a8d3537ba491f6f0c105fe9fdf1d13`

## Paired matrices

Pair 1, baseline → candidate:

| row | baseline | candidate | delta |
|---|---:|---:|---:|
| pp512 | 2337.6 | 2379.0 | +1.771% |
| pp2048 | 2415.8 | 2452.8 | +1.532% |
| pp8192 | 2322.5 | 2355.5 | +1.421% |
| pp32768 | 1944.1 | 1961.1 | +0.874% |
| decode | 36.5315 | 36.5550 | +0.064% |

Pair 2, candidate → baseline:

| row | candidate | baseline | delta |
|---|---:|---:|---:|
| pp512 | 2361.3 | 2365.9 | -0.194% |
| pp2048 | 2435.4 | 2433.2 | +0.090% |
| pp8192 | 2339.4 | 2318.7 | +0.893% |
| pp32768 | 1956.8 | 1945.3 | +0.591% |
| decode | 36.5076 | 36.5201 | -0.034% |

Mean paired pp8192 delta: **+1.157%**. This is below the superseding +1.4% admission threshold, and pair 2 pp512 is negative, so the candidate is killed.

## Exactness and compile screen

- Device oracle: all SET/ADD, gate/down/tails/K256 cases bit-identical; `GFX12-IU4 ORACLE PASS`.
- Full ADD/SET: 202 VGPR, 40 SGPR, zero VGPR/SGPR spills, zero scratch.
- Full-tile ADD epilogue: address VALU 448 → 2 per tile (7.0 → 0.03125/store); vector bounds compares 16 → 0; `v_cndmask` 0 → 0; two scalar tile-bound comparisons remain.
- Synthetic A/B/A gate proxy: 6803.105 / 6625.905 / 6816.367 us (+2.774% candidate throughput).
- Synthetic A/B/A residual proxy: 2988.959 / 2985.721 / 2989.541 us (+0.118% candidate throughput).

## Evidence

- `oracle.log`
- `isa-census.json`, `baseline.s`, `candidate-final.s`
- `probe-A1.log`, `probe-B.log`, `probe-A2.log`, `probe-summary.json`
- `daemon-md5.txt`
- `baseline-matrix.json`, `candidate-matrix.json`, `pair1-summary.json`
- `candidate-matrix-2.json`, `baseline-matrix-2.json`, `pair2-summary.json`
