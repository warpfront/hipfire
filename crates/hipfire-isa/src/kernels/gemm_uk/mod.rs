//! Zero-cost, composable WMMA GEMM fragments. Scheduling and register
//! allocation belong to the caller; these parts preserve per-output order.
pub mod tile;
pub mod chain;
pub mod prefetch;
pub mod epilogue;
pub mod sched;
pub use tile::{Tile, FragmentLayout, RegisterDirect, Lds};
pub use chain::{Chain, Iu4, Iu8, MmaKind};
pub use prefetch::Prefetch;
pub use epilogue::{DENSE_SILU_TEMPS, Epilogue};
