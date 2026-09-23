fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flags = rdna_compute::FeatureFlags::from_active_config(&args[1]);
    assert_eq!(flags.residual_ldsstage, args[2].parse::<bool>().unwrap());
    assert_eq!(flags.residual_ksplit_off, args[3].parse::<bool>().unwrap());
    println!("{}: ldsstage={}, kill={}", args[1], flags.residual_ldsstage, flags.residual_ksplit_off);
}
