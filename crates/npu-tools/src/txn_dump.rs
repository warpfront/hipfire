// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use std::collections::BTreeMap;
use std::fmt::Write;

fn word(data: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = data
        .get(offset..offset.checked_add(4).ok_or("transaction offset overflow")?)
        .ok_or_else(|| format!("truncated transaction at byte {offset}"))?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}
fn kind(row: u32, mem_rows: u8) -> &'static str {
    if row == 0 {
        "shim"
    } else if row < 1 + u32::from(mem_rows) {
        "mem"
    } else {
        "core"
    }
}
fn split(addr: u32) -> (u32, u32, u32) {
    ((addr >> 25) & 0x7f, (addr >> 20) & 0x1f, addr & 0xfffff)
}
pub fn decode(data: &[u8], summary: bool) -> Result<String, String> {
    let header = data.get(..16).ok_or("truncated transaction header")?;
    let num_ops = word(data, 8)?;
    let size = word(data, 12)?;
    let mut out = format!(
        "txn v{}.{} dev_gen={} rows={} cols={} mem_rows={} ops={num_ops} bytes={size}/{}\n",
        header[0],
        header[1],
        header[2],
        header[3],
        header[4],
        header[5],
        data.len()
    );
    let mut counts = BTreeMap::<(String, &'static str, u32), usize>::new();
    let mut off = 16usize;
    for _ in 0..num_ops {
        let op = word(data, off)? & 0xff;
        let w = |index| word(data, off + index * 4);
        let (name, row, offset, length, line) = match op {
            0 => {
                w(5)?;
                let (c, r, o) = split(w(2)?);
                (
                    "WRITE".into(),
                    r,
                    o,
                    24,
                    format!(
                        "WRITE  ({c},{r}) {:4} +{o:#07x} = {:#010x}",
                        kind(r, header[5]),
                        w(4)?
                    ),
                )
            }
            1 => {
                let n = w(3)? as usize;
                if n < 16 {
                    return Err(format!("invalid BLOCK length {n} at byte {off}"));
                }
                let (c, r, o) = split(w(2)?);
                let count = (n - 16) / 4;
                let mut line = format!(
                    "BLOCK  ({c},{r}) {:4} +{o:#07x} x{count} ",
                    kind(r, header[5])
                );
                for i in 0..count {
                    if i != 0 {
                        line.push(' ');
                    }
                    write!(line, "{:08x}", w(4 + i)?).unwrap();
                }
                ("BLOCK".into(), r, o, n, line)
            }
            3 | 4 => {
                w(6)?;
                let (c, r, o) = split(w(2)?);
                let name = if op == 3 { "MASKW" } else { "POLL" };
                let prefix = if op == 3 { "MASKW " } else { "POLL  " };
                (
                    name.into(),
                    r,
                    o,
                    28,
                    format!(
                        "{prefix} ({c},{r}) {:4} +{o:#07x} = {:#010x} mask {:#010x}",
                        kind(r, header[5]),
                        w(4)?,
                        w(5)?
                    ),
                )
            }
            0x80 => {
                let a = w(2)?;
                let b = w(3)?;
                let c = (a >> 16) & 0xff;
                let r = (a >> 8) & 0xff;
                (
                    "SYNC".into(),
                    r,
                    0,
                    w(1)? as usize,
                    format!(
                        "SYNC   ({c},{r}) dir={} chan={} ncol={} nrow={}",
                        a & 0xff,
                        (b >> 24) & 0xff,
                        (b >> 16) & 0xff,
                        (b >> 8) & 0xff
                    ),
                )
            }
            0x81 => {
                let (c, r, o) = split(w(6)?);
                let plus = u64::from(w(10)?) | (u64::from(w(11)?) << 32);
                (
                    "PATCH".into(),
                    r,
                    o,
                    w(1)? as usize,
                    format!(
                        "PATCH  ({c},{r}) {:4} +{o:#07x} arg={} plus={plus:#x}",
                        kind(r, header[5]),
                        w(8)?
                    ),
                )
            }
            _ => {
                let n = w(1)? as usize;
                (format!("OP{op:#x}"), 0, 0, n, format!("OP{op:#x} len={n}"))
            }
        };
        off = off
            .checked_add(length)
            .ok_or("transaction offset overflow")?;
        if summary {
            *counts
                .entry((
                    name,
                    if op == 0x80 {
                        "-"
                    } else {
                        kind(row, header[5])
                    },
                    if op == 0x80 { 0 } else { offset & 0xfff00 },
                ))
                .or_default() += 1;
        } else {
            writeln!(out, "{line}").unwrap();
        }
    }
    for ((name, tile, block), n) in counts {
        writeln!(out, "{n:6} {name:6} {tile:4} block {block:#07x}").unwrap();
    }
    Ok(out)
}
pub fn run(args: &[String]) -> Result<(), String> {
    let path = args
        .first()
        .ok_or("usage: npu-tools txn-dump INSTS.bin [--summary]")?;
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let text = decode(&data, args.iter().any(|s| s == "--summary"))?;
    use std::io::Write as _;
    std::io::stdout()
        .lock()
        .write_all(text.as_bytes())
        .map_err(|e| e.to_string())
}
