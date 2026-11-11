// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

#ifndef HIPFIRE_QSA_MROPE_POSITION_H
#define HIPFIRE_QSA_MROPE_POSITION_H

// The production HIP source and the GPU-free host test compile this exact
// implementation. There is no independent host mirror of the pooling rule.
#if defined(__HIPCC__) || defined(__HIP__)
#define QSA_POSITION_INLINE static __host__ __device__ __forceinline__
#else
#define QSA_POSITION_INLINE static inline
#endif

QSA_POSITION_INLINE int qsa_mrope_position(
    int position, int frequency, const int* positions, int prompt_rows,
    int delta, int section_h, int section_w) {
    if (positions == nullptr) return position;
    if (position >= prompt_rows) return position + delta;
    int axis = frequency % 3 == 1 && frequency < 3 * section_h ? 1
        : frequency % 3 == 2 && frequency < 3 * section_w ? 2 : 0;
    return positions[position * 3 + axis];
}

QSA_POSITION_INLINE int qsa_pooled_key_position(
    int first_block, int local_block, int compress, int frequency,
    const int* positions, int prompt_rows, int delta,
    int section_h, int section_w) {
    // Pooling averages several raw rows, but rotation uses the FIRST token
    // of the original block, including incremental pools at image boundaries.
    int block = first_block + local_block;
    int first_token = block * compress;
    return qsa_mrope_position(first_token, frequency, positions, prompt_rows,
        delta, section_h, section_w);
}

#undef QSA_POSITION_INLINE
#endif
