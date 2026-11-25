### Tests: Qwen4 sampled-restore twin

- **Added: `qwen4_kld restore`, the in-process twin of the conformance row
  `qwen4-ar-sampled-restore`.** It reproduces the daemon's legacy AR route:
  turn 1 is sampled from the request seed and committed live, and turn 2 is
  either restored from it or prefilled cold. It compares turn-2 decode logits
  (max|Δ|, full-vocab KL, argmax) and the CPU sampler's RNG state around each
  draw against the cold prefill. The control arms are a single-call prefill, a
  prefill split at the turn-1 boundary, and the live route's forwards without
  the session cache.
