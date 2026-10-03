/// True, logging the skip, when the pinned ROCm LLVM is absent (the no-GPU CI
/// runner). Its gates still run wherever the toolchain exists.
pub fn no_llvm() -> bool {
    const LLVM: &str = "/opt/rocm/core-10.0/lib/llvm/bin";
    let missing = !std::path::Path::new(LLVM).exists();
    if missing {
        eprintln!("skip: no {LLVM}");
    }
    missing
}
