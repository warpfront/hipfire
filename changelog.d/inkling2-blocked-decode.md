### Performance: Inkling-2-Mini decode through the blocked body at one row (gfx1201)

- **Faster: Inkling-2-Mini decode on gfx1201 typed-XTS loads (arch 24), about
  1.3x.** `generate_inkling` decodes through the new
  `inkling2::forward_decode_gpu`, which runs the blocked prefill body at one
  row (dense GEMV at M=1, routed experts as the default W4A8 grouped GEMMs)
  and keeps `forward_token_gpu` on every other load. A one-row block now
  attends through the split-KV FA2 decode kernels over gather maps uploaded
  once per call, instead of the blocked FA2 prefill kernel, which tiles queries
  only (8.2 ms to 1.2 ms of attention per token at 2K context). Per token on
  R9700: 4,477 to 2,656 launches, and 218 `hipMalloc` / 26 `hipFree` / 98
  `hipMemcpy` to 0 / 0 / 3. Daemon A/B against beta, 4 fresh loads per arm,
  64 greedy tokens: ctx 128 38.2 to 51.5 tok/s, ctx 2048 38.6 to 50.1 tok/s,
  same text. Paired decode-order KLD against the scalar decode body: wt2
  −1.3e-4 (95% CI below 0), code24 +9.6e-5 (95% CI includes 0).
- `inkling_kld` adds `eval --route decode-scalar|decode-blocked` (blocked
  prefill of the prompt rows, then one decode step per scored position) and
  `bench-decode` (decode speed at a given context, per body). Gate 3b gains
  `gate3b_multi_token_logits_decode` on the product decode step.
