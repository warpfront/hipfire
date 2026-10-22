# PR 940 validation evidence

The [current-head receipts](current-head/README.md) record the 2026-10-22 GPU numerical and product battery refresh after a normal merge with beta `88e9f0d6f`. The independent TP1/TP2 numerical gate is failed, not waived. Maintainer disposition and current-head performance remeasurement remain pending; historical throughput is not current-head proof.

The separately committed [historical raw QC](raw-qc/README.md) includes the explicitly authorized full generated answers, reasoning and original output JSON, with a [SHA256 manifest](raw-qc/manifest.json). It preserves the original beta `66db8cb5` provenance and the earlier length-limited runs. It excludes weights, executable binaries, raw F32 tensors and input dataset contexts. Reviewers can inspect this evidence-only commit without selecting it for mainline.
