// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

pub mod barriers;
pub mod cfg;
pub mod hazards;
mod kernarg_ix;
pub mod kernargs;
pub mod lds;
pub mod liveness;
pub(crate) mod predicates;
pub mod resources;
pub mod waits;
pub mod windows;
