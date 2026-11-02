// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use std::fmt::Write;

const CORE_BITS: &[&str] = &[
    "Enable",
    "Reset",
    "MemStall_S",
    "MemStall_W",
    "MemStall_N",
    "MemStall_E",
    "LockStall_S",
    "LockStall_W",
    "LockStall_N",
    "LockStall_E",
    "StreamStall_SS0",
    "b11",
    "StreamStall_MS0",
    "b13",
    "CascadeStall_SCD",
    "CascadeStall_MCD",
    "DebugHalt",
    "ECC_ErrorStall",
    "ECC_ScrubStall",
    "ErrorHalt",
    "CoreDone",
    "ProcBusStall",
];
const DMA_BITS: &[(u32, &str)] = &[
    (2, "LockAcqStall"),
    (3, "LockRelStall"),
    (4, "StreamStarved"),
    (5, "TCT/CountFull"),
    (8, "ErrLockUnavail"),
    (9, "ErrDMUnavail"),
    (10, "ErrBDUnavail"),
    (11, "ErrBDInvalid"),
    (12, "ErrFoTLength"),
    (13, "ErrFoTBDs"),
    (18, "TaskQOverflow"),
    (19, "Running"),
];
fn word(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}
fn core_status(value: u32) -> String {
    let mut text = String::new();
    for (i, name) in CORE_BITS.iter().enumerate() {
        if value & (1 << i) == 0 {
            continue;
        }
        if !text.is_empty() {
            text.push('|');
        }
        text.push_str(name);
    }
    if text.is_empty() {
        text.push('0');
    }
    text
}
fn dma(value: u32) -> String {
    if value == 0 {
        return "0".into();
    }
    let mut text = format!(
        "{} bd={} q={}",
        ["IDLE", "STARTING", "RUNNING", "?"][(value & 3) as usize],
        (value >> 24) & 0x3f,
        (value >> 20) & 7
    );
    let mut separator = ' ';
    for (bit, name) in DMA_BITS {
        if value & (1 << bit) == 0 {
            continue;
        }
        text.push(separator);
        text.push_str(name);
        separator = ',';
    }
    text
}
fn locks(data: &[u8]) -> String {
    let mut text = String::new();
    for (i, x) in data.iter().enumerate() {
        if i != 0 {
            text.push(' ');
        }
        write!(text, "{}", x & 0x3f).unwrap();
    }
    text
}
pub fn decode(data: &[u8], bitmap: Option<u32>) -> String {
    let count = data.len() / 504;
    let bitmap = bitmap.unwrap_or_else(|| {
        if count >= 32 {
            u32::MAX
        } else {
            (1u32 << count) - 1
        }
    });
    let mut out = String::new();
    for (i, col) in (0..32).filter(|col| bitmap & (1 << col) != 0).enumerate() {
        let Some(blk) = data.get(i * 504..(i + 1) * 504) else {
            break;
        };
        for row in 0..4 {
            let o = row * 80;
            let d: [u32; 4] = std::array::from_fn(|j| word(blk, o + j * 4));
            let st = word(blk, o + 48);
            let lock = &blk[o + 64..o + 80];
            if st == 2 && d.iter().all(|&v| v == 0) && lock.iter().all(|&v| v == 0) {
                continue;
            }
            writeln!(
                out,
                "core ({col},{}) status={st:#x} [{}] pc={:#x} sp={:#x} lr={:#x}",
                row + 2,
                core_status(st),
                word(blk, o + 52),
                word(blk, o + 56),
                word(blk, o + 60)
            )
            .unwrap();
            writeln!(
                out,
                "   s2mm0 {} | mm2s0 {} | s2mm1 {} | mm2s1 {}",
                dma(d[0]),
                dma(d[1]),
                dma(d[2]),
                dma(d[3])
            )
            .unwrap();
            let events = |start| {
                (0..4)
                    .map(|j| format!("{:08x}", word(blk, start + j * 4)))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            writeln!(
                out,
                "   locks {}  core_ev {}  mem_ev {}",
                locks(lock),
                events(o + 16),
                events(o + 32)
            )
            .unwrap();
        }
        let d: [u32; 12] = std::array::from_fn(|j| word(blk, 320 + j * 4));
        let lock = &blk[392..456];
        if d.iter().any(|&v| v != 0) || lock.iter().any(|&v| v != 0) {
            writeln!(out, "mem  ({col},1)").unwrap();
            for ch in 0..6 {
                if d[ch * 2] != 0 || d[ch * 2 + 1] != 0 {
                    writeln!(
                        out,
                        "   ch{ch}: s2mm {} | mm2s {}",
                        dma(d[ch * 2]),
                        dma(d[ch * 2 + 1])
                    )
                    .unwrap();
                }
            }
            let text = lock
                .iter()
                .enumerate()
                .filter(|(_, x)| **x != 0)
                .map(|(j, x)| format!("{j}:{}", x & 0x3f))
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(out, "   locks {text}").unwrap();
        }
        let d: [u32; 4] = std::array::from_fn(|j| word(blk, 456 + j * 4));
        let lock = &blk[488..504];
        if d.iter().any(|&v| v != 0) || lock.iter().any(|&v| v != 0) {
            writeln!(
                out,
                "shim ({col},0) s2mm0 {} | mm2s0 {} | s2mm1 {} | mm2s1 {}  locks {}",
                dma(d[0]),
                dma(d[1]),
                dma(d[2]),
                dma(d[3]),
                locks(lock)
            )
            .unwrap();
        }
    }
    out
}
fn parse_bitmap(text: &str) -> Result<u32, String> {
    let text = text.trim();
    let (negative, digits) = if let Some(s) = text.strip_prefix('-') {
        (true, s)
    } else {
        (false, text.strip_prefix('+').unwrap_or(text))
    };
    let prefix = digits.get(..2).unwrap_or("");
    let radix = match prefix {
        "0x" | "0X" => 16,
        "0b" | "0B" => 2,
        "0o" | "0O" => 8,
        _ => 10,
    };
    let prefixed = radix != 10;
    let body = if prefixed { &digits[2..] } else { digits };
    let mut value = 0u32;
    let mut previous_digit = false;
    let mut any_digit = false;
    let mut nonzero = false;
    for (i, ch) in body.chars().enumerate() {
        if ch == '_' && (previous_digit || (prefixed && i == 0)) {
            previous_digit = false;
            continue;
        }
        let digit = ch
            .to_digit(radix)
            .ok_or_else(|| format!("invalid bitmap: {text}"))?;
        value = value.wrapping_mul(radix).wrapping_add(digit);
        previous_digit = true;
        any_digit = true;
        nonzero |= digit != 0;
    }
    if !any_digit || !previous_digit || (!prefixed && digits.starts_with('0') && nonzero) {
        return Err(format!("invalid bitmap: {text}"));
    }
    Ok(if negative {
        value.wrapping_neg()
    } else {
        value
    })
}
pub fn run(args: &[String]) -> Result<(), String> {
    let path = args
        .first()
        .ok_or("usage: npu-tools aie-status DUMP.bin [BITMAP]")?;
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let bitmap = args.get(1).map(|s| parse_bitmap(s)).transpose()?;
    use std::io::Write as _;
    std::io::stdout()
        .lock()
        .write_all(decode(&data, bitmap).as_bytes())
        .map_err(|e| e.to_string())
}
