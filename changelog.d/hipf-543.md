### Fixes: Gemma4 EAGLE drafter arch mapping

- **Fixed: stock Gemma 4 drafters classify as arch 22.** Google's `google/gemma-4-*-it-assistant` configs (`model_type: gemma4_assistant`, `Gemma4AssistantForCausalLM`) were classified as arch 13, so `hipfire-quantize` wrote a target-arch file the drafter loader refused. Both strings now map to 22; `docs/MODELS.md` documents the `hipfire-quantize --format q8f16` drafter conversion (HIPF-543).
