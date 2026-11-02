; hipcc regions of gdn_chunk_prep (kernels/src/gdn_chunk_scan_prep.gfx1201.hip md5 1ecb5a312c9a7b962421a7424cee641b),
; hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress -ffp-contract=off (the rdna-compute recipe for this module),
; AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961).
; Lifted with peacemaker-lift from /home/kaden/qcal/perf/fp8-4k5/gdnprep-fuse/kill/prep_cache.hsaco (sha256 b2b10a2fd241fcdef8c6309799e3d5e0204d9efa6fad189d8ac0a694f50e501a); region `cvt_v`.
; Regenerate/compare: `peacemaker-isa region-import --gdn-object <object>` (feature `lift`).
; inputs: o0=v5 o1=v4 o2=v3 o3=v2
; outputs: h0=v5.l h1=v5.h h2=v3.l h3=v48.l
v_cvt_f16_f32_e32 v3.l, v3
v_cvt_f16_f32_e32 v5.l, v5
v_cvt_f16_f32_e32 v5.h, v4
v_cvt_f16_f32_e32 v48.l, v2
