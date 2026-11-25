### Fixes: Inkling-2-Mini decode after the first token

- **Fixed: Inkling-2-Mini GPU decode diverged from the CPU reference after the
  first token (arch 24).** Two bugs in `hipfire-arch-inkling2`'s GPU forward,
  neither visible to the token-0 parity gate. The MoE post-conv history sat at
  column `4 * n_embd_kv` instead of `2 * n_embd_kv + n_embd`, overlapping the
  last 256 attention-output history channels. Loop passes 1 and 2 also built
  their messages from every block of the previous pass instead of only the
  blocks after the prelude and the first core block. New ignored gate
  `gate3b_multi_token_logits` in `tests/gpu_parity.rs` compares GPU decode
  with the CPU cached decode at positions 0..3.
