//! NT4 vs NT8 wide expert-run tile byte oracle (A3B geometry, gfx12 PeaceMaker
//! builder entries). Wrapper: the oracle lives in
//! `moe_iu4_wide_tiles_oracle/lab.rs`, a child module of the `qwen4_moe_sym`
//! example whose private oracle machinery it reuses, so this file includes that
//! example by path. See the lab module docs for modes, output and admission.
//!
//!   cargo run -p rdna-compute --features lab --example moe_iu4_wide_tiles_oracle -- --plan
//!   cargo run -p rdna-compute --features lab --example moe_iu4_wide_tiles_oracle -- --self-check
//!   cargo run --release -p rdna-compute --features lab --example moe_iu4_wide_tiles_oracle

#[path = "qwen4_moe_sym.rs"]
#[allow(dead_code, unused_imports)]
mod sym;

fn main() {
    std::process::exit(sym::wide_tiles::run());
}
