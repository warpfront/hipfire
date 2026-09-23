# ProducerBody state after KFD wedge

- Base: `bc1a55de998c91d064304b4e6d2db95fa4bcd2c0`
- Device used: card-D UUID `GPU-6109a4cb5f833235`; HOME `ab3`.
- Bundle is not committed. Exact tracked diff: `/home/kaden/ClaudeCode/warpfront/wt-prodbody/scratch-2026-09-17/ProducerBody/bundle-uncommitted.diff`.
- Modified tracked file: `kernels/src/block_i4_128_quant.hip`.

## Paired matrix (`--pp 512,8192 --ctx 128 --tg 128`, GDN ON)

- Pair 1 base -> bundle: pp8192 2735.30 -> 2905.50 tok/s, +6.22%; pp512 2660.30 -> 2793.90, +5.02%; decode 31.3581 -> 31.3589 tok/s, +0.0025%.
- Pair 2 bundle -> base: pp8192 2692.30 -> 2897.50 tok/s (base -> bundle), +7.62%; pp512 2634.70 -> 2785.50, +5.72%; decode 31.3884 -> 31.3527 tok/s, -0.114%.

## First TTFT pair

- Base median 2285.451 ms.
- Bundle median 2461.266 ms.
- Bundle slower by 7.693%.
- Raw: `/home/kaden/ClaudeCode/warpfront/wt-prodbody/scratch-2026-09-17/ProducerBody/ttft-base.log`, `/home/kaden/ClaudeCode/warpfront/wt-prodbody/scratch-2026-09-17/ProducerBody/ttft-bundle.log`.

## KLD

- c2 OFF 0.060029, ON 0.059423 (ON-OFF -0.000606).
- c24 OFF 0.081170, ON 0.080450 (ON-OFF -0.000720).
- Output md5s: `{"c24_off": "651c2407b4074bf935d9977bc1a8ede4", "c24_on": "ef8c6fb1c7c1f9a6379fccafdb435b63", "c2_off": "8ee9a5f97862532b0482ccd442fbca94", "c2_on": "67a9ac8901f56b0cbd57db88cd24d26a"}`.
- The earlier 0.079153 GDN-ON pin used the original eight-candidate quant source (SHA256 `df39676b...`); this bundle uses the four-candidate source (SHA256 `9c0b07b0...`). Both logs show `prefill_chunk: requested=8192 admitted=8192`, so chunk 4096 vs 8192 is not the cause of the different absolute KLD.

## Remaining after reboot

1. Reversed TTFT pair (bundle then base). The first attempt wedged KFD; the retry could not initialize HIP.
2. Candidate serve battery, require 5/5 healthy turns.
3. Reconcile TTFT pairs; ship only if final gate passes.
4. Commit the tracked source with measured numbers and KLD output md5s if all gates pass.

Do not start GPU work until the reboot completes.
