// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware hooks of the exact BF16 epilogue: the frozen DAG `mq_i4_reference::grouped_prefill_epilogue_with` run over
//! twin F32 sums with the three device intrinsics (`v_exp_f32`, `v_ldexp_f32`, IEEE divide expansion) answered by the
//! compiled real-G1 probe code object (never `PortableEpilogue`, no portable-exp fallback). Same replay protocol and
//! divide transcription self-check as `qwen4_moe_sym_real_g1`.

use std::cell::{Cell, RefCell};
use std::path::Path;

use hip_bridge::{Function, HipRuntime, Module};
use hipfire_runtime::mq_i4_reference::{self, DecodeKind, EpilogueIntrinsics};
use serde_json::{json, Value};

use super::common::*;

pub struct Probe {
    pub exp2: Function,
    pub ldexp: Function,
    pub div: Function,
    pub ka: [usize; 3],
}

pub struct Hook<'a> {
    pub hip: &'a HipRuntime,
    pub probe: Probe,
    _module: Module,
    pub div_checked: Cell<u64>,
    pub div_mismatch: Cell<u64>,
    pub div_ieee_mismatch: Cell<u64>,
    pub div_list: RefCell<Vec<Value>>,
    pub dag_passes: Cell<u64>,
    pub dag_calls: Cell<u64>,
}

/// Bitwise equality, any two NaNs equal (the payload/sign of an invalid-operation NaN is not part of the contract).
fn same_f32_bits(a: u32, b: u32) -> bool {
    a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

/// Correctly rounded (RNE, denormals preserved) IEEE-754 quotient bits, computed on the CPU.
fn ieee_div_bits(num: u32, den: u32) -> u32 {
    (f32::from_bits(num) / f32::from_bits(den)).to_bits()
}

struct Replay<'a> {
    prev: &'a [Vec<(u32, u32)>; 3],
    prev_ans: &'a [Vec<u32>; 3],
    cur: RefCell<[Vec<(u32, u32)>; 3]>,
    miss: Cell<bool>,
}
impl Replay<'_> {
    fn call(&self, k: usize, a: u32, b: u32) -> f32 {
        let mut cur = self.cur.borrow_mut();
        let pos = cur[k].len();
        cur[k].push((a, b));
        if pos < self.prev[k].len() && self.prev[k][pos] == (a, b) {
            f32::from_bits(self.prev_ans[k][pos])
        } else {
            self.miss.set(true);
            0.0
        }
    }
}
impl EpilogueIntrinsics for Replay<'_> {
    fn exp2(&self, x: f32) -> f32 {
        self.call(0, x.to_bits(), 0)
    }
    fn ldexp(&self, x: f32, e: i32) -> f32 {
        self.call(1, x.to_bits(), e as u32)
    }
    fn divide(&self, numerator: f32, denominator: f32) -> f32 {
        self.call(2, numerator.to_bits(), denominator.to_bits())
    }
}

fn probe_launch(
    hip: &HipRuntime,
    f: &Function,
    ka: usize,
    bufs: &[&hip_bridge::DeviceBuffer],
    n: usize,
) -> Result<()> {
    let off = bufs.len() * 8;
    if ka < off + 4 {
        return Err(format!(
            "probe kernarg size {ka} < explicit size {}",
            off + 4
        ));
    }
    let mut a = vec![0u8; ka];
    for (i, b) in bufs.iter().enumerate() {
        a[i * 8..i * 8 + 8].copy_from_slice(&(b.as_ptr() as u64).to_le_bytes());
    }
    a[off..off + 4].copy_from_slice(&(n as u32).to_le_bytes());
    // SAFETY: explicit args laid out per the probe kernels' (pointer.., i32 n) ABI; blob sized to the code object's
    // `.kernarg_segment_size`; every buffer outlives the synchronize below.
    unsafe {
        hip.launch_kernel_blob(
            f,
            [n.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            None,
            &mut a,
        )
    }
    .map_err(err)?;
    hip.device_synchronize().map_err(err)
}

impl<'a> Hook<'a> {
    pub fn load(
        hip: &'a HipRuntime,
        probe_img: &[u8],
        probe_path: &Path,
        probe_ka_override: Option<usize>,
    ) -> Result<Hook<'a>> {
        let module = hip.module_load_data(probe_img).map_err(err)?;
        let ka = |sym: &str| -> Result<usize> {
            match (metadata_kernarg_size(probe_img, sym), probe_ka_override) {
                (Some(m), Some(o)) if m != o => Err(format!(
                    "{sym}: kernarg metadata {m} != --probe-kernarg-bytes {o}"
                )),
                (Some(m), _) => Ok(m),
                (None, Some(o)) => Ok(o),
                (None, None) => Err(format!(
                    "{sym}: `.kernarg_segment_size` not found in {}; pass --probe-kernarg-bytes",
                    probe_path.display()
                )),
            }
        };
        let getf = |name: &str| {
            hip.module_get_function(&module, name)
                .map_err(|e| format!("{name}: {e}"))
        };
        let probe = Probe {
            exp2: getf("g1_probe_exp2")?,
            ldexp: getf("g1_probe_ldexp")?,
            div: getf("g1_probe_div")?,
            ka: [
                ka("g1_probe_exp2")?,
                ka("g1_probe_ldexp")?,
                ka("g1_probe_div")?,
            ],
        };
        Ok(Hook {
            hip,
            probe,
            _module: module,
            div_checked: Cell::new(0),
            div_mismatch: Cell::new(0),
            div_ieee_mismatch: Cell::new(0),
            div_list: RefCell::new(Vec::new()),
            dag_passes: Cell::new(0),
            dag_calls: Cell::new(0),
        })
    }

    fn eval_exp2(&self, x: &[u32]) -> Result<Vec<u32>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        let hip = self.hip;
        let (din, dout) = (up(hip, &u32s_le(x))?, hip.malloc(x.len() * 4).map_err(err)?);
        probe_launch(
            hip,
            &self.probe.exp2,
            self.probe.ka[0],
            &[&din, &dout],
            x.len(),
        )?;
        let r = le_u32s(&read_back(hip, &dout, x.len() * 4)?);
        free_all(hip, vec![din, dout])?;
        Ok(r)
    }

    fn eval_ldexp(&self, x: &[u32], e: &[u32]) -> Result<Vec<u32>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        let hip = self.hip;
        let (dx, de, dout) = (
            up(hip, &u32s_le(x))?,
            up(hip, &u32s_le(e))?,
            hip.malloc(x.len() * 4).map_err(err)?,
        );
        probe_launch(
            hip,
            &self.probe.ldexp,
            self.probe.ka[1],
            &[&dx, &de, &dout],
            x.len(),
        )?;
        let r = le_u32s(&read_back(hip, &dout, x.len() * 4)?);
        free_all(hip, vec![dx, de, dout])?;
        Ok(r)
    }

    /// `(explicit v_div_scale/rcp/fma/fmas/fixup sequence, hipcc __fdiv_rn)`.
    fn eval_div(&self, num: &[u32], den: &[u32]) -> Result<(Vec<u32>, Vec<u32>)> {
        if num.is_empty() {
            return Ok((vec![], vec![]));
        }
        let hip = self.hip;
        let n = num.len();
        let (dn, dd) = (up(hip, &u32s_le(num))?, up(hip, &u32s_le(den))?);
        let (d_seq, d_rn) = (
            hip.malloc(n * 4).map_err(err)?,
            hip.malloc(n * 4).map_err(err)?,
        );
        probe_launch(
            hip,
            &self.probe.div,
            self.probe.ka[2],
            &[&dn, &dd, &d_seq, &d_rn],
            n,
        )?;
        let r = (
            le_u32s(&read_back(hip, &d_seq, n * 4)?),
            le_u32s(&read_back(hip, &d_rn, n * 4)?),
        );
        free_all(hip, vec![dn, dd, d_seq, d_rn])?;
        Ok(r)
    }

    /// Known-answer test of the divide probes against CPU IEEE division (explicit v_div_* sequence, hipcc `__fdiv_rn`
    /// and the exact quotient agree bit for bit; includes den in `[2^126, inf]`, denormals, zero, inf, 65536 random pairs).
    pub fn divide_known_answer(&self) -> Result<(Value, bool)> {
        let nums: [f32; 22] = [
            -87.5,
            87.5,
            -88.7,
            -88.72,
            -87.0,
            -80.0,
            -1.0,
            1.0,
            0.5,
            1e-30,
            -1e-30,
            1e-40,
            3.0e38,
            -3.0e38,
            f32::MIN_POSITIVE,
            0.0,
            -0.0,
            65536.0,
            1e-20,
            4.0,
            1.0e30,
            f32::MAX,
        ];
        let mut dens: Vec<u32> = (0..64u32).map(|i| 0x7e80_0000 + i * 0x0004_0000).collect();
        dens.extend([
            0x7f80_0000,
            0x7f7f_ffff,
            0x7f7f_fffe,
            0x7e7f_ffff,
            0x7e80_0001,
            1.0f32.to_bits(),
            1.0f32.to_bits() + 1,
            0x0000_0001,
            0x0080_0000,
            0x0000_7fff,
            3.0f32.to_bits(),
            1.0e30f32.to_bits(),
            65536.0f32.to_bits(),
            0.5f32.to_bits(),
            0x8000_0000,
            0x7fc0_0000,
        ]);
        let r = 1.0017680e38f32.to_bits();
        dens.extend([r - 1, r, r + 1]);
        let (mut num, mut den): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        for n in nums {
            for d in &dens {
                num.push(n.to_bits());
                den.push(*d);
            }
        }
        let mut rng = Rng(0xD1_71DE_0000_0001);
        for _ in 0..65536 {
            let w = rng.next();
            num.push(w as u32);
            den.push((w >> 32) as u32);
        }
        let (seq, rn) = self.eval_div(&num, &den)?;
        let (mut seq_ieee, mut rn_ieee, mut seq_rn) = (0u64, 0u64, 0u64);
        let mut first: Vec<Value> = Vec::new();
        for i in 0..num.len() {
            let ieee = ieee_div_bits(num[i], den[i]);
            let (a, b, c) = (
                !same_f32_bits(seq[i], ieee),
                !same_f32_bits(rn[i], ieee),
                !same_f32_bits(seq[i], rn[i]),
            );
            seq_ieee += a as u64;
            rn_ieee += b as u64;
            seq_rn += c as u64;
            if (a || b || c) && first.len() < 64 {
                first.push(json!({"num": format!("{:#010x}", num[i]), "den": format!("{:#010x}", den[i]), "probe_sequence": format!("{:#010x}", seq[i]), "hipcc_fdiv_rn": format!("{:#010x}", rn[i]), "cpu_ieee": format!("{ieee:#010x}")}));
            }
        }
        let ok = seq_ieee == 0 && rn_ieee == 0 && seq_rn == 0;
        Ok((
            json!({"cases": num.len(), "probe_sequence_vs_cpu_ieee_mismatches": seq_ieee, "hipcc_fdiv_rn_vs_cpu_ieee_mismatches": rn_ieee, "probe_sequence_vs_fdiv_rn_mismatches": seq_rn,
            "first_mismatches": first, "ok": ok}),
            ok,
        ))
    }

    /// The frozen BF16 epilogue DAG over `sums` with the three device intrinsics. The intrinsic calls depend on each
    /// other, so the DAG is replayed until a pass answers every call from a device result for exactly the requested inputs.
    pub fn dag_epilogue(
        &self,
        kind: DecodeKind,
        m: usize,
        sums: &[f32],
        slot_ids: &[i32],
    ) -> Result<Vec<u16>> {
        let mut prev: [Vec<(u32, u32)>; 3] = Default::default();
        let mut prev_ans: [Vec<u32>; 3] = Default::default();
        let mut prev_rn: Vec<u32> = Vec::new();
        for _ in 0..8 {
            let rp = Replay {
                prev: &prev,
                prev_ans: &prev_ans,
                cur: RefCell::new(Default::default()),
                miss: Cell::new(false),
            };
            let out = mq_i4_reference::grouped_prefill_epilogue_with(kind, m, sums, slot_ids, &rp);
            let miss = rp.miss.get();
            let cur = rp.cur.into_inner();
            self.dag_passes.set(self.dag_passes.get() + 1);
            if !miss {
                self.dag_calls.set(
                    self.dag_calls.get() + (cur[0].len() + cur[1].len() + cur[2].len()) as u64,
                );
                for i in 0..prev_ans[2].len() {
                    self.div_checked.set(self.div_checked.get() + 1);
                    let (nu, de, seq) = (prev[2][i].0, prev[2][i].1, prev_ans[2][i]);
                    let ieee = ieee_div_bits(nu, de);
                    let seq_ieee_bad = !same_f32_bits(seq, ieee);
                    if seq_ieee_bad {
                        self.div_ieee_mismatch.set(self.div_ieee_mismatch.get() + 1);
                    }
                    if seq != prev_rn[i] || seq_ieee_bad {
                        if seq != prev_rn[i] {
                            self.div_mismatch.set(self.div_mismatch.get() + 1);
                        }
                        let mut l = self.div_list.borrow_mut();
                        if l.len() < 256 {
                            l.push(json!({"num": format!("{nu:#010x}"), "den": format!("{de:#010x}"), "probe_sequence": format!("{seq:#010x}"), "hipcc_fdiv_rn": format!("{:#010x}", prev_rn[i]), "cpu_ieee": format!("{ieee:#010x}"),
                                "num_f32": f32::from_bits(nu), "den_f32": f32::from_bits(de)}));
                        }
                    }
                }
                return Ok(out);
            }
            let a0 = self.eval_exp2(&cur[0].iter().map(|p| p.0).collect::<Vec<_>>())?;
            let a1 = self.eval_ldexp(
                &cur[1].iter().map(|p| p.0).collect::<Vec<_>>(),
                &cur[1].iter().map(|p| p.1).collect::<Vec<_>>(),
            )?;
            let (a2, rn) = self.eval_div(
                &cur[2].iter().map(|p| p.0).collect::<Vec<_>>(),
                &cur[2].iter().map(|p| p.1).collect::<Vec<_>>(),
            )?;
            prev = cur;
            prev_ans = [a0, a1, a2];
            prev_rn = rn;
        }
        Err("epilogue DAG replay did not converge in 8 passes".into())
    }
}

/// A loaded PM E8P module: the shipped gate/up and down entries and (optionally) their `_acc` F32-accumulator twins.
pub struct E8pModule {
    _module: Module,
    pub ship: [Function; 2],
    pub acc: Option<[Function; 2]>,
    /// `.kernarg_segment_size` of each shipped entry: exactly the 64-byte ABI (60 bytes + `n_experts@60`), enforced at load.
    pub ka: [usize; 2],
    pub syms: [String; 2],
    pub bind: Value,
    pub image_sha: String,
}

pub fn load_e8p(
    hip: &HipRuntime,
    co_path: &Path,
    img: &[u8],
    pinned: Option<&str>,
    bundle: Option<&Path>,
    syms: [String; 2],
    need_acc: bool,
) -> Result<E8pModule> {
    let bind = bind_module("e8p", co_path, img, pinned, bundle)?;
    let module = hip.module_load_data(img).map_err(err)?;
    let getf = |name: &str| {
        hip.module_get_function(&module, name)
            .map_err(|e| format!("{name}: {e}"))
    };
    let ship = [getf(&syms[0])?, getf(&syms[1])?];
    let acc = if need_acc {
        Some([
            getf(&format!("{}_acc", syms[0]))?,
            getf(&format!("{}_acc", syms[1]))?,
        ])
    } else {
        None
    };
    let kernarg_64 = |sym: &str| match metadata_kernarg_size(img, sym) {
        Some(64) => Ok(64),
        other => Err(format!(
            "{sym}: e8p kernarg segment {other:?}, expected the 64-byte ABI (n_experts@60)"
        )),
    };
    let ka = [kernarg_64(&syms[0])?, kernarg_64(&syms[1])?];
    Ok(E8pModule {
        _module: module,
        ship,
        acc,
        ka,
        syms,
        bind,
        image_sha: sha(img),
    })
}
