fn main() {
    let cfg = hipfire_config::load_local_process_config().unwrap();
    println!("IU8 legacy_value = {:?}", cfg.legacy_value("HIPFIRE_IU8_PREFILL"));
    println!("IU4 legacy_value = {:?}", cfg.legacy_value("HIPFIRE_IU4_PREFILL"));
    let flags = rdna_compute::FeatureFlags::from_process_config("gfx1201", &cfg);
    println!("iu8_prefill_enabled = {}", flags.iu8_prefill_enabled());
    println!("iu4_prefill_enabled = {}", flags.iu4_prefill_enabled());
}
