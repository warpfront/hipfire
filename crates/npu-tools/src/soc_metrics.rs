// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use std::io::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// `gpu_metrics` of the Halo iGPU (`railgun::npu::fclk::igpu_pci`, env `NPU_IGPU_PCI`).
pub fn path() -> String {
    format!(
        "/sys/bus/pci/devices/{}/gpu_metrics",
        railgun::npu::fclk::igpu_pci()
    )
}
pub const COLS: &str = "real_ns sysclk dram_rd dram_wr ipu_rd ipu_wr ipu_busy0 ipu_busy1 ipu_busy2 ipu_busy3 ipu_busy4 ipu_busy5 ipu_busy6 ipu_busy7 gfx_busy ipu_mW socket_mW gfxclk fclk ipuclk mpipuclk uclk";
/// Decode format 3.0, naturally aligned 264-byte SMU table. The caller supplies
/// realtime so offline fixtures and live sampling use exactly the same schema.
pub fn decode(data: &[u8], real_ns: u64) -> Result<[u64; 22], String> {
    if data.len() < 4 {
        return Err(format!("truncated gpu_metrics header len {}", data.len()));
    }
    if data[2] != 3 || data[3] != 0 || data.len() != 264 {
        return Err(format!(
            "unsupported gpu_metrics format {}.{} len {}",
            data[2],
            data[3],
            data.len()
        ));
    }
    let u16_at = |off| u64::from(u16::from_le_bytes(data[off..off + 2].try_into().unwrap()));
    let mut values = [0u64; 22];
    values[0] = real_ns;
    values[1] = u64::from_le_bytes(data[104..112].try_into().unwrap());
    for i in 0..4 {
        values[2 + i] = u16_at(94 + i * 2);
    }
    for i in 0..8 {
        values[6 + i] = u16_at(46 + i * 2);
    }
    values[14] = u16_at(42);
    values[15] = u16_at(116);
    values[16] = u64::from(u32::from_le_bytes(data[112..116].try_into().unwrap()));
    values[17] = u16_at(174);
    values[18] = u16_at(182);
    values[19] = u16_at(180);
    values[20] = u16_at(188);
    values[21] = u16_at(186);
    Ok(values)
}
pub fn format(values: &[u64; 22]) -> String {
    use std::fmt::Write as _;
    let mut line = String::with_capacity(180);
    for (i, value) in values.iter().enumerate() {
        if i != 0 {
            line.push(' ');
        }
        write!(line, "{value}").unwrap();
    }
    line
}
fn sample() -> Result<[u64; 22], String> {
    let path = path();
    let data = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    let real_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let real_ns = u64::try_from(real_ns).map_err(|_| "realtime timestamp exceeds u64")?;
    decode(&data, real_ns)
}
pub fn run(args: &[String]) -> Result<(), String> {
    let once = args.iter().any(|s| s == "--once");
    if once {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "{COLS}").map_err(|e| e.to_string())?;
        writeln!(out, "{}", format(&sample()?)).map_err(|e| e.to_string())?;
        return Ok(());
    }
    let mut out: Box<dyn Write> = if let Some(path) = args.first() {
        Box::new(std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?)
    } else {
        Box::new(std::io::stdout())
    };
    let period = args
        .get(1)
        .map(|s| {
            s.parse::<f64>()
                .map_err(|e| format!("invalid sample period: {e}"))
        })
        .transpose()?
        .unwrap_or(0.05);
    let period =
        Duration::try_from_secs_f64(period).map_err(|e| format!("invalid sample period: {e}"))?;
    writeln!(out, "{COLS}").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;
    loop {
        writeln!(out, "{}", format(&sample()?)).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
        std::thread::sleep(period);
    }
}
