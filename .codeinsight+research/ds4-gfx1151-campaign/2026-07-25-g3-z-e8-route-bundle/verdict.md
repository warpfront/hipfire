# G3.z — E8 launch bundle

Decision: reject as performance-neutral; retain shipping route v3.

The bundle combined the two exact E8 candidates whose isolated route-shape
projections cleared the campaign's 2% admission floor only in aggregate:

- 105 same-input E8 projection pairs were concatenated into one launch without
  sharing activation values or changing per-row arithmetic.
- The separate 43-call grouped O-LoRA projection used the exact four-group
  load-ahead schedule measured in G3.y.

The combined projection was 0.772 ms/token, or 2.055% of the 37.55 ms route.
The authoritative model result did not translate proportionally:

- fixture: 2,048 prompt tokens, 510 generated tokens, batch 1, temperature 0,
  coherent prose, top-k 6, KV depth 2,048--2,558
- order: ABBAAB, three fresh processes per arm
- route-v3 baseline: 27.60, 27.38, 27.63 tok/s; median 27.60
- candidate bundle: 27.79, 27.63, 27.67 tok/s; median 27.67
- median delta: +0.07 tok/s, +0.254%
- exact 729-resample 95% interval: 0.00 to +0.41 tok/s, 0.00% to +1.497%
- implied saving: 0.0917 ms/token, only 11.9% of the projected 0.772 ms

All six decoded outputs were byte-identical
(`65e473d239598dd9c90257996c426c9e`). The candidate retained route was fully
certified: 2,215 launches, 33 symbols, sequence hash
`12445125801925253362`, 57,035 command dwords, and 2,214/2,214 covered
boundaries. Shipping route v3 remained 2,320 launches, 32 symbols, sequence
hash `4393256763546932634`, 57,746 command dwords, and 2,319/2,319 covered
boundaries.

The first attempted acceptance session is excluded. Its first baseline process
JIT-compiled the baseline grouped-E8 code object, changing the kernel-cache
fingerprint; the harness correctly stopped before run 2. The accepted session
used the stable cache hash
`8252ae151883bb0bcdd88b154080e6083b310b48142e17c37a806208d5b5017e`
for every process.

Evidence:

- `/home/kaden/ds4-gfx1151-evidence/2026-07-25-g3-z-e8-route-bundle/20260725T155112Z-87b9d5f1`
- acceptance binary SHA-256:
  `514ba505106ad06eb1f03c1b86270260329bf6c0993214e217bc6bf735599dc1`

Skipped:

- no route version bump or promotion
- no credit from the short 2,048/128 warm screen
- no cross-function 43-pair extension after the bundle failed to translate
- no artifact, tensor-map, dense-bpw, sidecar, NPU, or MFP2 work
