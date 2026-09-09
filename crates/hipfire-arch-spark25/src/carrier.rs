// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Loader-facing bundle construction for Spark-X2.5.
//!
//! Maple-style thin surface: this crate MUST NOT depend on `hipfire_loader`
//! (circular). The real `Carrier` impl lives in `hipfire-loader` and calls
//! [`load_spark25_bundle`]. Public HFQ entry is
//! [`crate::bundle::load_spark25_from_hfq`].

use crate::bundle::Spark25Bundle;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Build the Spark GPU bundle from a loader `ModelSource`.
///
/// Re-exported at crate root as the stable loader entry point.
pub fn load_spark25_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Spark25Bundle, String> {
    crate::bundle::load_spark25_bundle(src, ctx)
}
