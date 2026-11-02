// SPDX-License-Identifier: Apache-2.0
//! Device side of the oracle: weight residency arms (VRAM, device-mapped pinned host, guarded VMM end), raw module
//! launches with the frozen 64-byte ABI (`n_experts@60` = the pointer table's actual entry count), output poisoning and tail guards, and the code-object denorm audit.

use crate::model::*;
use hip_bridge::{DeviceBuffer, Function, HipRuntime, VmmArena, HIP_HOST_MALLOC_MAPPED};
use serde_json::{json, Value};
use std::ffi::c_void;

pub const TAIL: usize = 4096;

// ---------------------------------------------------------------------------------------------------------------
// Code-object denorm audit (kernel descriptor COMPUTE_PGM_RSRC1: FLOAT_DENORM_MODE_32 bits 16..17, _16_64 18..19)
// ---------------------------------------------------------------------------------------------------------------

fn rd16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn rd32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn rd64(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}

fn elf_kd_rsrc1(elf: &[u8], kernel: &str) -> Option<u32> {
    if elf.len() < 64 || &elf[..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return None;
    }
    let shoff = rd64(elf, 0x28)? as usize;
    let shentsize = rd16(elf, 0x3a)? as usize;
    let shnum = rd16(elf, 0x3c)? as usize;
    let sh = |i: usize| -> Option<(u32, u64, u64, u64, u32)> {
        let o = shoff + i * shentsize;
        Some((
            rd32(elf, o + 4)?,
            rd64(elf, o + 16)?,
            rd64(elf, o + 24)?,
            rd64(elf, o + 32)?,
            rd32(elf, o + 40)?,
        ))
    };
    let want = format!("{kernel}.kd");
    for i in 0..shnum {
        let (ty, _, off, size, link) = sh(i)?;
        if ty != 2 && ty != 11 {
            continue; // SHT_SYMTAB, SHT_DYNSYM
        }
        let (_, _, stroff, strsize, _) = sh(link as usize)?;
        for s in 0..(size as usize / 24) {
            let so = off as usize + s * 24;
            let nm = rd32(elf, so)? as usize;
            let shndx = rd16(elf, so + 6)? as usize;
            let val = rd64(elf, so + 8)?;
            let names = elf.get(stroff as usize + nm..(stroff + strsize) as usize)?;
            let end = names.iter().position(|&c| c == 0)?;
            if &names[..end] == want.as_bytes() {
                let (_, addr, soff, _, _) = sh(shndx)?;
                let fo = soff as usize + val.checked_sub(addr)? as usize;
                return rd32(elf, fo + 48);
            }
        }
    }
    None
}

/// `(fp32 denorm mode, fp16/64 denorm mode)` of `kernel` in a (possibly offload-bundled) code object.
pub fn kernel_denorm(image: &[u8], kernel: &str) -> Res<(u32, u32)> {
    let mut i = 0usize;
    while i + 4 <= image.len() {
        if &image[i..i + 4] == b"\x7fELF" {
            if let Some(r) = elf_kd_rsrc1(&image[i..], kernel) {
                return Ok(((r >> 16) & 3, (r >> 18) & 3));
            }
        }
        i += 1;
    }
    Err(format!(
        "kernel descriptor `{kernel}.kd` not found in any embedded ELF"
    ))
}

// ---------------------------------------------------------------------------------------------------------------
// Residency arms
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm {
    Vram,
    /// `hipHostMalloc(mapped)` pages read by the GPU through their device alias.
    Mapped,
    /// VMM mapping whose last tensor ends exactly at the mapped end; the following VA is reserved but unmapped.
    Guarded,
}

impl Arm {
    pub const ALL: [Arm; 3] = [Arm::Vram, Arm::Mapped, Arm::Guarded];
    pub fn name(self) -> &'static str {
        match self {
            Arm::Vram => "vram",
            Arm::Mapped => "mapped_host_alias",
            Arm::Guarded => "vmm_guarded_end",
        }
    }
}

fn align(n: usize) -> usize {
    n.div_ceil(SLAB_ALIGN) * SLAB_ALIGN
}

/// Tensors at 256-byte aligned offsets with 0xFF sentinels between (and after, unless `trailing == false`).
fn layout(tensors: &[&[u8]], trailing: bool) -> (Vec<u8>, Vec<usize>) {
    let mut buf = Vec::new();
    let mut offs = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        offs.push(buf.len());
        buf.extend_from_slice(t);
        if trailing || i + 1 < tensors.len() {
            buf.resize(align(buf.len()) + SLAB_ALIGN, 0xff);
        }
    }
    (buf, offs)
}

pub struct Resident {
    pub arm: Arm,
    pub ptrs: Vec<u64>,
    pub info: Value,
    vram: Option<DeviceBuffer>,
    host: Option<*mut c_void>,
    arena: Option<VmmArena>,
}

impl Resident {
    pub fn new(hip: &HipRuntime, dev: i32, arm: Arm, tensors: &[&[u8]]) -> Res<Self> {
        match arm {
            Arm::Vram => {
                let (buf, offs) = layout(tensors, true);
                let b = hip.malloc(buf.len().max(8)).map_err(err)?;
                hip.memcpy_htod(&b, &buf).map_err(err)?;
                let base = b.as_ptr() as u64;
                let info = json!({"arm": arm.name(), "base": format!("{base:#x}"), "bytes": buf.len(), "trailing_sentinel": true});
                Ok(Self {
                    arm,
                    ptrs: offs.iter().map(|o| base + *o as u64).collect(),
                    info,
                    vram: Some(b),
                    host: None,
                    arena: None,
                })
            }
            Arm::Mapped => {
                let (buf, offs) = layout(tensors, true);
                let host = hip
                    .host_malloc(buf.len(), HIP_HOST_MALLOC_MAPPED)
                    .map_err(err)?;
                // SAFETY: `host` is a live hipHostMalloc allocation of exactly `buf.len()` bytes.
                unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), host as *mut u8, buf.len()) };
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                let dev_ptr = hip.host_get_device_pointer(host, 0).map_err(err)?;
                if dev_ptr.is_null() {
                    let _ = hip.host_free(host);
                    return Err("hipHostGetDevicePointer returned null".into());
                }
                let base = dev_ptr as u64;
                let info = json!({"arm": arm.name(), "host_ptr": format!("{:#x}", host as u64), "device_alias": format!("{base:#x}"),
                    "alias_differs_from_host_ptr": base != host as u64, "bytes": buf.len(), "trailing_sentinel": true});
                Ok(Self {
                    arm,
                    ptrs: offs.iter().map(|o| base + *o as u64).collect(),
                    info,
                    vram: None,
                    host: Some(host),
                    arena: None,
                })
            }
            Arm::Guarded => {
                let (buf, offs) = layout(tensors, false);
                let len = buf.len();
                let req = len.div_ceil(2 << 20) * (2 << 20) + (4 << 20);
                let mut arena = VmmArena::reserve(hip, dev, req).map_err(err)?;
                let gran = arena.granularity().max(1);
                let mapped = len.div_ceil(gran) * gran;
                let setup = (|| -> Res<(Vec<u64>, Value)> {
                    if arena.reserved_bytes() < mapped + gran {
                        return Err(format!("VMM reserve {} leaves no unmapped guard granule after mapped {mapped} (granularity {gran})", arena.reserved_bytes()));
                    }
                    let front = mapped - len;
                    if front % SLAB_ALIGN != 0 {
                        return Err(format!("guarded-end layout: front pad {front} is not a multiple of {SLAB_ALIGN} (last tensor length not 256-aligned)"));
                    }
                    arena.map_next(hip, mapped, &[dev]).map_err(err)?;
                    let view = arena.buffer(mapped).map_err(err)?;
                    let mut data = vec![0xffu8; front];
                    data.extend_from_slice(&buf);
                    hip.memcpy_htod_offset(&view, 0, &data).map_err(err)?;
                    let base = arena.base_address() as u64;
                    let ptrs: Vec<u64> = offs.iter().map(|o| base + (front + *o) as u64).collect();
                    let last_end = ptrs.last().copied().unwrap_or(base)
                        + tensors.last().map(|t| t.len() as u64).unwrap_or(0);
                    if last_end != base + mapped as u64 {
                        return Err(
                            "guarded-end layout: last tensor does not end at the mapped end".into(),
                        );
                    }
                    let info = json!({"arm": arm.name(), "base": format!("{base:#x}"), "mapped_bytes": mapped, "granularity": gran,
                        "reserved_bytes": arena.reserved_bytes(), "unmapped_guard_va_bytes": arena.reserved_bytes() - mapped,
                        "last_tensor_end_equals_mapped_end": true, "last_tensor_bytes": tensors.last().map(|t| t.len()).unwrap_or(0),
                        "trailing_sentinel": false});
                    Ok((ptrs, info))
                })();
                match setup {
                    Ok((ptrs, info)) => Ok(Self {
                        arm,
                        ptrs,
                        info,
                        vram: None,
                        host: None,
                        arena: Some(arena),
                    }),
                    Err(e) => {
                        let _ = arena.release(hip);
                        Err(e)
                    }
                }
            }
        }
    }

    /// Free after the caller has synchronized.
    pub fn release(mut self, hip: &HipRuntime) -> Res<()> {
        if let Some(b) = self.vram.take() {
            hip.free(b).map_err(err)?;
        }
        if let Some(h) = self.host.take() {
            hip.host_free(h).map_err(err)?;
        }
        if let Some(mut a) = self.arena.take() {
            a.release(hip).map_err(err)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------------------------------------------

/// Per-scenario device inputs shared by every launch (candidate, anchor, negatives, every arm).
pub struct DeviceCase<'a> {
    pub hip: &'a HipRuntime,
    pub kind: Kind,
    pub tiles_n: usize,
    pub m_total: usize,
    pub div: usize,
    pub rows: usize,
    pub tiles_dev: DeviceBuffer,
    pub sorted_dev: DeviceBuffer,
    pub x_dev: DeviceBuffer,
}

pub fn upload(hip: &HipRuntime, bytes: &[u8]) -> Res<DeviceBuffer> {
    let b = hip.malloc(bytes.len().max(8)).map_err(err)?;
    hip.memcpy_htod(&b, bytes).map_err(err)?;
    Ok(b)
}

impl<'a> DeviceCase<'a> {
    pub fn new(
        hip: &'a HipRuntime,
        kind: Kind,
        tiles: &[Tile],
        div: usize,
        rows: usize,
        xq: &[u8],
    ) -> Res<Self> {
        let tile_ids: Vec<u8> = tiles.iter().flat_map(|t| t.0.to_le_bytes()).collect();
        let sorted: Vec<u8> = tiles
            .iter()
            .flat_map(|t| t.1)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Ok(Self {
            hip,
            kind,
            tiles_n: tiles.len(),
            m_total: tiles.len() * 16,
            div,
            rows,
            tiles_dev: upload(hip, &tile_ids)?,
            sorted_dev: upload(hip, &sorted)?,
            x_dev: upload(hip, xq)?,
        })
    }

    pub fn free(self) -> Res<()> {
        self.hip.free(self.tiles_dev).map_err(err)?;
        self.hip.free(self.sorted_dev).map_err(err)?;
        self.hip.free(self.x_dev).map_err(err)
    }

    /// Launch `f` over `table` (the device `u64` expert-pointer table; its `n_experts` kernarg is its actual entry count)
    /// into a poisoned output of `body_bytes` plus a poisoned tail. Returns the body bytes and the number of tail bytes that changed.
    pub fn run(
        &self,
        f: &Function,
        table: &DeviceBuffer,
        body_bytes: usize,
    ) -> Res<(Vec<u8>, u64)> {
        self.run_multi(&[(f, table, &self.tiles_dev)], body_bytes)
    }

    /// Sequential launches (each with its own expert-pointer table and tile-id array, e.g. per-codec masks) into ONE
    /// initially poisoned output, with the shared slot array, activations and the 64-byte ABI (`n_experts@60` = each
    /// table's actual entry count). Synchronizes after each launch.
    pub fn run_multi(
        &self,
        launches: &[(&Function, &DeviceBuffer, &DeviceBuffer)],
        body_bytes: usize,
    ) -> Res<(Vec<u8>, u64)> {
        let hip = self.hip;
        let y = hip.malloc(body_bytes + TAIL).map_err(err)?;
        hip.memset(&y, 0xa5, body_bytes + TAIL).map_err(err)?;
        for (f, table, tiles) in launches {
            let mut a = [0u8; 64];
            for (off, p) in [
                (0usize, table.as_ptr() as u64),
                (8, tiles.as_ptr() as u64),
                (16, self.sorted_dev.as_ptr() as u64),
                (24, self.x_dev.as_ptr() as u64),
                (32, y.as_ptr() as u64),
            ] {
                a[off..off + 8].copy_from_slice(&p.to_le_bytes());
            }
            for (off, v) in [
                (40usize, self.kind.m() as u32),
                (44, self.kind.k() as u32),
                (48, self.div as u32),
                (52, self.m_total as u32),
                (56, self.rows as u32),
                (60, (table.size() / 8) as u32),
            ] {
                a[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
            // SAFETY: exact 64-byte ABI (60 bytes + trailing n_experts); every buffer outlives the synchronize below.
            unsafe {
                hip.launch_kernel_blob(
                    f,
                    [self.kind.grid_x(), self.tiles_n as u32, 1],
                    [self.kind.threads(), 1, 1],
                    0,
                    None,
                    &mut a,
                )
            }
            .map_err(err)?;
            hip.device_synchronize().map_err(err)?;
        }
        let mut all = vec![0u8; body_bytes + TAIL];
        hip.memcpy_dtoh(&mut all, &y).map_err(err)?;
        hip.free(y).map_err(err)?;
        let tail_bad = all[body_bytes..].iter().filter(|&&b| b != 0xa5).count() as u64;
        all.truncate(body_bytes);
        Ok((all, tail_bad))
    }

    pub fn acc_bytes(&self) -> usize {
        self.m_total * self.kind.m() * 4
    }
    pub fn h_bytes(&self) -> usize {
        self.m_total * self.kind.out_cols() * 2
    }
}

/// Upload the u64 pointer table, run, free it.
pub fn with_ptr_table<T>(
    hip: &HipRuntime,
    ptrs: &[u64],
    f: impl FnOnce(&DeviceBuffer) -> Res<T>,
) -> Res<T> {
    let bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
    let b = upload(hip, &bytes)?;
    let r = f(&b);
    hip.free(b).map_err(err)?;
    r
}
