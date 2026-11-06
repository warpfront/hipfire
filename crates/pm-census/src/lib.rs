// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! pm-census: developer tooling of the PM/railgun bedrock (R2).
//!
//! * [`rocprof`] — rocprofv3 trace reader (JSON and CSV), owned data only.
//! * [`inventory`] — object inventory of the repo's embedded PM objects and
//!   hipcc kernel names, and the per-dispatch origin join.
//! * [`census`] — time-weighted PM / hipcc / railgun attribution of one
//!   profiled ROI, and the model x arch x phase matrix (unmeasured cells
//!   are explicit, never zero).
//! * [`extract`] — `WorkloadCaseV1` corpus extraction from real routes
//!   (rocprofv3 traces, Redline captures, railgun program manifests,
//!   pm_loop records).
//! * [`ledger`] — ingest of the retained-PM4 dispatch ledger
//!   (`HIPFIRE_REDLINE_LEDGER`) into a count- or timestamp-weighted
//!   `CensusV1` cell for the dispatches rocprofv3 cannot see.
//! * [`noise`] — measurement-noise harness over pm_loop records: A/A null
//!   controls, per-round ABBA dispersion, cross-process sign tests.
//! * [`model`] — model descriptor: `config.json` integers and the `.hfq`
//!   tensor directory (header only) as weight groups.
//! * [`hw`] — per-arch peak table for rooflines.
//! * [`ingest`] — `pm-census extract nodes` / `roofline` / `queue`: one
//!   trace + model descriptor -> node corpus through `KernelAbiV1` + `WorkV1`.
//! * [`capture`] — `pm-census extract route-hist` / `act-capture`: captured MoE
//!   route dumps and activation rows -> sha256-identified `route_hist.v1` /
//!   `act_capture.v1` judge inputs.

pub mod capture;
pub mod census;
pub mod extract;
pub mod hw;
pub mod ingest;
pub mod inventory;
pub mod ledger;
pub mod model;
pub mod noise;
pub mod rocprof;
pub mod tune;
