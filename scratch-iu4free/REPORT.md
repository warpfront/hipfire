# Fold-free iu4 prefill experiment

## Result

The artifact quality gate passes, and a one-accumulator exponent-shift kernel is viable but not ready to ship. On the paired card-C standalone screen it improves the gate/up shape by 10.58%, is effectively flat on down (-0.59%), and improves the two-shape aggregate by 7.14%. The environment-gated implementation is retained for follow-up; the default path is unchanged.

Daemon gates were not run because card-B was occupied by the row-global-A gate and the fold-free candidate had not completed the required qkv family screens. Therefore there are no defensible tg128, pp8192, TTFT, or daemon GEMM-us/token numbers for this candidate.

## Quality gates

All scores use the exact 24-chunk prefill KLD evaluator on card-C unless noted.

| artifact / activation route | c2 | c24 | result |
|---|---:|---:|---|
| pow2h, existing K128 activation scales | 0.071897 | **0.091338** | pass (`<= 0.10`) |
| shipped control, existing K128 activation scales | 0.064864 | 0.081199 | control |
| pow2h, one activation scale per token | 0.046252 | 0.055408 | pass |
| full-pow2, one activation scale per token | 0.058773 | **0.071194** | pass |
| pow2h row-max normalized weights, one activation scale per token | 0.098325 | 0.117126 | fail |
| pow2h MSE-selected row-common weights, one activation scale per token | not run | 0.114720 | fail |
| pow2h mantissa collapsed to integer weights | 0.121697 | not run | fail early |

The full-pow2 plus row-global-activation route is the clean numeric contract for shift folding. The pre-enlargement candidate produced byte-identical c2 and c24 score binaries to that route. After the tile enlargement and metadata fix, the rebuilt candidate again scored c2 = 0.058773 and its score binary was byte-identical to the baseline route.

## Weight screens

A row-max common-scale requantization changed 5,821,518,243 of 25,621,954,560 codes (22.72%) with weight relative RMS error 0.085655. It failed c24 at 0.117126 even after row-global activation quantization.

An MSE-selected row-common scale changed 5,640,816,401 codes with relative RMS error 0.090104 and still failed c24 at 0.114720. Removing the half-step mantissa changed 7,577,507,991 codes with relative RMS error 0.12933 and failed c2 at 0.121697. A fully common weight scale is therefore not quality-admissible without retraining.

For the full-pow2 artifact, all 4,152,320 rows have exponent span at most seven. Span counts are `{0: 1,988,637, 1: 2,145,437, 2: 17,488, 3: 651, 4: 90, 5: 9, 6: 7, 7: 1}`. Adjacent K128 exponent changes are overwhelmingly zero or one, but their direction differs by row, so a wave cannot use one uniform shift direction.

## Kernel

The experimental route is selected only by `HIPFIRE_IU4_SHIFT_FOLDFREE=1`; the normal dispatch is unchanged. `HIPFIRE_IU4_GLOBAL_A_SCREEN=1` selects the one-scale-per-token activation quantizer, with fused producer quantization disabled for screening.

The kernel:

- keeps one integer accumulator set;
- uses a 256-token by 128-row workgroup tile with eight waves;
- retains a 64-by-64 tile per wave;
- shifts live accumulators between adjacent K128 exponent domains;
- applies one final floating-point scale after the K loop;
- uses 30,720 bytes LDS.

Resource compile for the full-set entry point: 184 VGPR, 44 SGPR, zero scratch, zero spills. Runtime occupancy reports two blocks/CU with 30,720-byte LDS.

A first 128-token/four-wave geometry reached 206.538 TOPS on gate/up versus 225.916 TOPS baseline and was discarded. Enlarging the token tile restored active waves and made gate/up positive. A metadata publisher bug in the first enlarged measurements let threads 128-255 read rows outside the 128-row tile; final measurements below are from the corrected kernel, where only the first 128 threads publish row metadata.

## Corrected standalone gate

Card-C, B=8192, five warmups plus 20 HIP-event samples, median of positions 9 and 10:

| shape | baseline | shift-fold | delta |
|---|---:|---:|---:|
| gate/up, M=34816 K=5120 | 12,910.419 us, 226.219 TOPS | 11,675.602 us, 250.144 TOPS | **+10.58%** |
| down, M=5120 K=17408 | 5,148.720 us, 283.622 TOPS | 5,179.470 us, 281.938 TOPS | **-0.59%** |
| two-shape aggregate | 18,059.139 us, 242.584 TOPS | 16,855.072 us, 259.914 TOPS | **+7.14%** |

For these two GEMMs across 64 layers, the standalone timing corresponds to 141.087 us/token baseline and 131.680 us/token candidate. This is not a daemon profile and must not be presented as total production GEMM time.

## Missing production gates

The qkvza and qkv shape screens, card-B daemon warmup/pair, tg128, pp8192, TTFT, and daemon profiler attribution were not completed. No shipping claim is made. The candidate is experimental, off by default, and requires those gates before integration.

## Reproduction pointers

- `pow2h-iu4-c24.log`: initial admissibility gate, 0.091338.
- `pow2-global-a-iu4-c24.log`: full-pow2 row-global route, 0.071194.
- `rownorm-opt-global-a-c24.log`: row-common MSE screen, 0.114720.
- `bench-baseline-corrected-pair.txt` and `bench-shiftfold-corrected-gate.txt`: gate/up pair.
- `bench-baseline-corrected-down-pair.txt` and `bench-shiftfold-corrected-down.txt`: down pair.
