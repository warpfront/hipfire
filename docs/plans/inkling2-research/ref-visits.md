# Inkling-2-Mini reference forward + per-visit hidden dumps

Status: DONE 2026-11-21. Validation PASS against the llama.cpp oracle.
Branch/worktree untouched (`crates/*` not modified); only this doc + the
research script + dumps were created.

## What was built

`D:\hipfire-models\ref-visits\inkling2_ref_forward.py` — single-token, pos-0,
float32 PyTorch port of the merged llama.cpp forward
(`src/models/inkling2.cpp` + `inkling2-loop.cpp` @ `29b52d398b`), cross-checked
op-by-op against transformers `modeling_inkling2.py` (installed 5.23.0 has the
base `Inkling2*` layers but NOT the loop wiring; the loop schedule is
llama.cpp-only). Companion bisection probe: `probe_compare.py` (runs
transformers `Inkling2SlidingAttention` with bin-loaded weights side-by-side
with the reference; all projections/convs/clips/norms agree to 0.0).

Semantics implemented (all verified against BOTH sources where they overlap):
controller eps 1e-6 + tanh gates, `mix()` (RMSNorm per state, 1/sqrt(2304),
cur appended last), per-head-vector L2 `clip()` (eps 1e-6, factor clamp
[0,1]), unified short-conv state with post-scale `1/sqrt(5)` static /
`1/sqrt(9)` dynamic, relative-bias gather with sink subtracted pre-gather,
tau (exactly 1.0 at pos 0), MoE sigmoid top-8 + bias-selection, renorm
eps 1e-20 then `* route_scale 8`, SiTU(4, 25), GQA **block** grouping
(head `h` shares kv head `h//3`), pre-mix MoE residual base, final
`output_res_query` mix + norm + F32 head + softcap 30 + unpadded slice.

## Weight source (no 104 GB rehash)

- `D:\hipfire-models\Inkling-2-Mini\inkling2-f32.gguf` (104293518144 B) → exported
  once to `D:\hipfire-models\gguf-f32.bin` (104281569826 B) by
  `C:\Users\Jan\AppData\Local\Temp\opencode\gguf_to_f32_bin.py`.
- Tensor layout rule (verified structurally + by oracle match): the bin holds
  the converter's torch tensor bytes; **torch view = `flat.reshape(*reversed(bin_dims))`**.
  e.g. `attn_q (2304,3072)→(3072,2304)`, experts `(2304,2304,128)→(128,2304,2304)`,
  rel table `(16,1024)→(1024,16)`, `loop_msg_query (2304,4)→(4,2304)` rows [Q,K,V,FFN].
- Conv tap convention (converted domain): static torch-view `[C,4]`, tap index **3
  = current** (matches ggml `ssm_conv` cross-correlation + the converter's
  `flip(0).T`); dynamic proj rows pack as `row = lag + head*4`.

## Prompt / tokens

- Golden prompt g1 from `D:\hipfire-models\inkling2-golden\MANIFEST.txt`:
  `The capital of France is`, md5 `1aacd3c5c05cf9695cc799acc59581938d` (matches MANIFEST).
- Tokenizer (`D:\hipfire-models\Inkling-2-Mini`, no BOS): `[371, 151451, 149879, 151194, 149884]`.
- Dumped: token id **0** (ties to existing `oracle-logits-tok0.txt`) and token id
  **371** (first token of g1), each single-token decode at pos 0.

## Validation (vs `D:\hipfire-models\oracle-logits-tok0.txt`, 200058 logits)

`max|d| = 1.85e-03`, `mean|d| = 4.18e-04`, argmax 31274 == 31274, Pearson r = 1.0000.
Residual is torch-CPU vs ggml-CPU summation order over 48 visits; use
`atol=1e-2` for early visits, looser later, plus expert-selection equality.

## Dumps (`D:\hipfire-models\ref-visits\`)

Per run dir (`tok0/`, `tok371/`): `visit00.npy`…`visit47.npy` (float32 `[2304]`,
= layer output feeding the next visit), `final_hidden.npy` (post final-norm),
`logits.npy` (`[200058]` float32), `experts.json` (per-visit top-8 ids + weights).
File list + md5: `md5sums.txt` (102 files). Key hashes:

```text
9765c2624a61cbab160ca930d9eb1c26  tok0/logits.npy
eea6cdbec9dd9cbca6bbe2614b41da8d  tok0/final_hidden.npy
35a3c232dedd4474338ca2496a40307d  tok371/logits.npy
9e416df05d5c4aa5b237ce6441ee11bf  tok371/final_hidden.npy
```

`run_meta.json` records bin path/size, prompt md5, token ids, validation numbers.

## How the Rust side should compare

Instrument `forward_token` (worker branch `crates/hipfire-arch-inkling2/src/forward.rs`):
record `x` after each `decode_visit` (visits 0..47), the post-`final_norm` hidden,
and `lm_head_logits`. Compare visit-by-visit against `tok371/visitNN.npy` (and
`tok0/` for the already-dumped oracle token); compare top-8 sets against
`experts.json`. First visit whose hidden diverges (or whose expert set flips)
localizes the bug. NOTE: this reference reads the **F32 GGUF** weights, so the
Rust comparison must use the `GgufF32` path — but read the warnings below first.

## Findings for the worker (source-reading, worker to confirm by measurement)

1. **GQA mapping (PROVEN by this validation).** `forward.rs`/`decode_visit` and the
   old `decode_layer` use kv head `h % 8`. ggml flash-attn maps `ik2 = iq2 / rk2`
   (`ops.cpp`, integer division) and HF `repeat_kv` uses block grouping: head `h`
   shares kv head `h // 3`. My reference reproduces the oracle ONLY with `h//3`.
   This alone decorrelates every layer.
2. **Clip granularity.** `oracle::clip` (and its use in `decode_visit`) takes one
   norm over the whole `[n_head*d]` vector. llama.cpp `clip()` norms per
   head-vector (`ggml_sum_rows`), as does transformers `Inkling2{L2,ScaledL2}Clip`
   (`vector_norm(dim=-1)`). Diverges at visit 0.
3. **MoE residual base.** `forward.rs` returns `ffn_in(mixed+controlled) + moe_conv`.
   llama.cpp adds the MoE output onto the pre-mix `cur` (`residual = cur`,
   `return residual + moe_c` in the reference port of `build_moe_ffn`): the base
   is the visit input, not `ffn_in(mixed+controlled)`. Diverges wherever `mix()`
   moves the stream; compare visit 0 expert sets first.
4. **Message duplication / bank.** Reference banks the previous pass's
   `blocks[prelude_len+1..]` plus its output and restores the prelude each pass
   (skip-first-core-block detail from `inkling2-loop.cpp`). A forward that banks
   all blocks or forgets the restore double-counts block 6 output from pass 2 on.
5. **Missing output mix.** Reference applies the final `output_res_query` mix +
   norm before the F32 head (softcap 30, unpadded slice to 200058). A scaffold
   that goes straight from visit-47 output to `lm_head` will match hiddens but
   miss logits.
6. **Deferred dynamic taps + single-distance gather (known gaps, not bugs).**
   Reference implements static convs; full-layer (5/11/17/23) input-dependent
   `weight_proj` taps (`row = lag + head*4`) and multi-distance rel gather are
   the documented next steps. Expect residual divergence on full layers even
   after 1-5 are fixed; use `probe_compare.py` (transformers layer vs ref,
   agrees to 0.0 through value-norm) to bisect.
7. **`GgufF32` layout hazard.** The `.bin` carries file-order dim labels with
   torch-order bytes (`flat.reshape(*reversed(bin_dims))`). The shim is correct
   only where it ignores labels (plain linears, `loop_msg_query` reassembly,
   clips) and scrambles controllers, experts, and full-layer `query_rows`
   (see `inkling2-tensor-mapping.md` §6.2/§6.4). Compare Rust hiddens via the
   HFQ/Raw path, not `GgufF32`, until the shim reverses bin dims at read.

(Reconstructed 2026-11-21: the worker's file ended mid-sentence at finding 3
with a literal `...[truncated 2085 chars]` marker; findings 3-7 above are
restored from `inkling2_ref_forward.py` comments, `probe_compare.py`,
`run_meta.json`, and the tensor-mapping traps. Validation numbers in §Validation
are untouched.)