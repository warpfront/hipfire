### Tests: dense Q8 attention conformance fixture and CPU reference

- **Added: a seeded Q8F16 LLaMA-class HFQM and an independent CPU reference.**
  `hipfire-arch-llama` gains `fixture` (a 1.7 MB, 2-layer, byte-deterministic HFQM
  that the production carrier loads with every weight in `Q8F16`) and
  `cpu_reference` (own weight decode and a teacher-forced `f64` forward). The
  `dense_q8_anchor_gpu` test compares engine logits with the reference on three
  fixed prompts against a frozen budget, and rejects a planted wrong dequant
  scale. The fixture digest is pinned in `fixtures/pins.toml`.
