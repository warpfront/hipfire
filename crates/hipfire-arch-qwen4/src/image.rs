// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash Next uses the Qwen image processor: RGB normalized by mean/std 0.5,
//! patch size 16, temporal duplication 2 and spatial merge 2. Reuse the same
//! decode, resource bounds, resize and grouped patch layout as Qwen3.5-VL.

pub use hipfire_arch_qwen35_vl::image::{
    extract_patches, load_and_preprocess, load_and_preprocess_from_bytes, smart_resize,
};
