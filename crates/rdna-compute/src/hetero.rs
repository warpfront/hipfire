//! Launchers for the device-side hetero expert split (`kernels/src/hetero_split.hip`).
//!
//! The mailbox lives in pinned host memory (`hipfire_dispatch::hetero_mailbox`
//! owns the layout); its fields are passed as raw device addresses, which equal
//! the host addresses on both GPUs of the tested pair.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

const MODULE: &str = "hetero_split";

/// Device addresses of one layer's mailbox slot fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeteroSlotPtrs {
    pub req_seq: u64,
    pub resp_seq: u64,
    pub error: u64,
    pub x_rot: u64,
    pub ids: u64,
    pub weights: u64,
    pub cold_out: u64,
}

fn p<T>(v: &T) -> *mut c_void {
    v as *const T as *mut c_void
}

impl Gpu {
    fn hetero_launch(
        &mut self,
        func: &str,
        grid: u32,
        block: u32,
        params: &mut [*mut c_void],
        blob: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(MODULE, kernels::HETERO_SPLIT_SRC, func)?;
        self.launch_maybe_blob(func, [grid, 1, 1], [block, 1, 1], 0, params, blob)
    }

    /// `counter[0] += 1` (one thread): the first step of a decode token.
    pub fn hetero_token_advance(&mut self, counter: &GpuTensor) -> HipResult<()> {
        let c = counter.buf.as_ptr();
        let mut params = vec![p(&c)];
        self.hetero_launch("hetero_token_advance", 1, 1, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(c);
            b
        })
    }

    /// Publish one layer's route into its mailbox slot and write the primary's
    /// masked ids (cold experts -> -1), then release the request sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn hetero_publish_cold(
        &mut self,
        counter: &GpuTensor,
        layer: u32,
        layers: u32,
        x_rot: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
        cold_bitmap: &GpuTensor,
        n_experts: i32,
        rows: i32,
        top_k: i32,
        dim: i32,
        slot: &HeteroSlotPtrs,
        masked_ids: &GpuTensor,
    ) -> HipResult<()> {
        let (c, x, i, w, bm, m) = (
            counter.buf.as_ptr(),
            x_rot.buf.as_ptr(),
            ids.buf.as_ptr(),
            weights.buf.as_ptr(),
            cold_bitmap.buf.as_ptr(),
            masked_ids.buf.as_ptr(),
        );
        let (mx, mi, mw, rq) = (
            slot.x_rot as *mut c_void,
            slot.ids as *mut c_void,
            slot.weights as *mut c_void,
            slot.req_seq as *mut c_void,
        );
        let mut params = vec![
            p(&c), p(&layer), p(&layers), p(&x), p(&i), p(&w), p(&bm), p(&n_experts),
            p(&rows), p(&top_k), p(&dim), p(&mx), p(&mi), p(&mw), p(&m), p(&rq),
        ];
        self.hetero_launch("hetero_publish_cold", 1, 256, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(c);
            b.push_u32(layer);
            b.push_u32(layers);
            b.push_ptr(x);
            b.push_ptr(i);
            b.push_ptr(w);
            b.push_ptr(bm);
            b.push_i32(n_experts);
            b.push_i32(rows);
            b.push_i32(top_k);
            b.push_i32(dim);
            b.push_ptr(mx);
            b.push_ptr(mi);
            b.push_ptr(mw);
            b.push_ptr(m);
            b.push_ptr(rq);
            b
        })
    }

    /// Bounded wait (one thread) for the layer's response; on timeout the slot's
    /// error word gets the expected sequence.
    pub fn hetero_await_cold(
        &mut self,
        counter: &GpuTensor,
        layer: u32,
        layers: u32,
        slot: &HeteroSlotPtrs,
        max_polls: u32,
    ) -> HipResult<()> {
        let c = counter.buf.as_ptr();
        let (rs, er) = (slot.resp_seq as *mut c_void, slot.error as *mut c_void);
        let mut params = vec![p(&c), p(&layer), p(&layers), p(&rs), p(&er), p(&max_polls)];
        self.hetero_launch("hetero_await_cold", 1, 1, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(c);
            b.push_u32(layer);
            b.push_u32(layers);
            b.push_ptr(rs);
            b.push_ptr(er);
            b.push_u32(max_polls);
            b
        })
    }

    /// Copy the cold slots' rows from the mailbox into `down_expanded` (one
    /// block per routed slot); a no-op after a timed-out wait.
    #[allow(clippy::too_many_arguments)]
    pub fn hetero_install_cold(
        &mut self,
        slot: &HeteroSlotPtrs,
        ids: &GpuTensor,
        cold_bitmap: &GpuTensor,
        n_experts: i32,
        slots: u32,
        dim: i32,
        down_expanded: &GpuTensor,
    ) -> HipResult<()> {
        let (er, co) = (slot.error as *mut c_void, slot.cold_out as *mut c_void);
        let (i, bm, d) = (ids.buf.as_ptr(), cold_bitmap.buf.as_ptr(), down_expanded.buf.as_ptr());
        let mut params = vec![p(&er), p(&co), p(&i), p(&bm), p(&n_experts), p(&dim), p(&d)];
        self.hetero_launch("hetero_install_cold", slots, 256, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(er);
            b.push_ptr(co);
            b.push_ptr(i);
            b.push_ptr(bm);
            b.push_i32(n_experts);
            b.push_i32(dim);
            b.push_ptr(d);
            b
        })
    }

    /// Expert device: masked ids for the cold kernels (hot experts -> -1) from
    /// the mailbox ids at `mb_ids`.
    pub fn hetero_mask_hot(
        &mut self,
        mb_ids: u64,
        cold_bitmap: &GpuTensor,
        n_experts: i32,
        slots: i32,
        masked_ids: &GpuTensor,
    ) -> HipResult<()> {
        let mi = mb_ids as *mut c_void;
        let (bm, m) = (cold_bitmap.buf.as_ptr(), masked_ids.buf.as_ptr());
        let mut params = vec![p(&mi), p(&bm), p(&n_experts), p(&slots), p(&m)];
        self.hetero_launch("hetero_mask_hot", 1, 256, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(mi);
            b.push_ptr(bm);
            b.push_i32(n_experts);
            b.push_i32(slots);
            b.push_ptr(m);
            b
        })
    }

    /// Expert device, last in a layer's graph: `resp_seq = req_seq` after a
    /// system fence.
    pub fn hetero_store_seq(&mut self, req_seq: u64, resp_seq: u64) -> HipResult<()> {
        let (rq, rs) = (req_seq as *mut c_void, resp_seq as *mut c_void);
        let mut params = vec![p(&rq), p(&rs)];
        self.hetero_launch("hetero_store_seq", 1, 1, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(rq);
            b.push_ptr(rs);
            b
        })
    }

    /// Prefill: convert a chunk's normed rows to F16 in `x_f16` (local staging
    /// for the copy across the link), write the route into the prefill slot at
    /// `mb_ids` / `mb_w`, and the primary's masked ids (cold -1).
    #[allow(clippy::too_many_arguments)]
    pub fn hetero_publish_rows(
        &mut self,
        x_norm: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
        cold_bitmap: &GpuTensor,
        n_experts: i32,
        rows: i32,
        top_k: i32,
        dim: i32,
        x_f16: &GpuTensor,
        mb_ids: u64,
        mb_w: u64,
        masked_ids: &GpuTensor,
    ) -> HipResult<()> {
        let (x, i, w, bm, xf, m) = (
            x_norm.buf.as_ptr(),
            ids.buf.as_ptr(),
            weights.buf.as_ptr(),
            cold_bitmap.buf.as_ptr(),
            x_f16.buf.as_ptr(),
            masked_ids.buf.as_ptr(),
        );
        let (mi, mw) = (mb_ids as *mut c_void, mb_w as *mut c_void);
        let mut params = vec![
            p(&x), p(&i), p(&w), p(&bm), p(&n_experts), p(&rows), p(&top_k), p(&dim), p(&xf), p(&mi), p(&mw), p(&m),
        ];
        let grid = ((rows.max(1) as u32 * dim as u32).div_ceil(256)).clamp(1, 1024);
        self.hetero_launch("hetero_publish_rows", grid, 256, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(x);
            b.push_ptr(i);
            b.push_ptr(w);
            b.push_ptr(bm);
            b.push_i32(n_experts);
            b.push_i32(rows);
            b.push_i32(top_k);
            b.push_i32(dim);
            b.push_ptr(xf);
            b.push_ptr(mi);
            b.push_ptr(mw);
            b.push_ptr(m);
            b
        })
    }

    /// Prefill, one sub-chunk, after its rows were copied: write the layer,
    /// first row and row count at `meta` (three u32), then release `req_seq`.
    #[allow(clippy::too_many_arguments)]
    pub fn hetero_release_rows(
        &mut self,
        counter: &GpuTensor,
        layer: u32,
        layers: u32,
        row_off: i32,
        rows: i32,
        meta: u64,
        req_seq: u64,
    ) -> HipResult<()> {
        let c = counter.buf.as_ptr();
        let (me, rq) = (meta as *mut c_void, req_seq as *mut c_void);
        let mut params = vec![p(&c), p(&layer), p(&layers), p(&row_off), p(&rows), p(&me), p(&rq)];
        self.hetero_launch("hetero_release_rows", 1, 1, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(c);
            b.push_u32(layer);
            b.push_u32(layers);
            b.push_i32(row_off);
            b.push_i32(rows);
            b.push_ptr(me);
            b.push_ptr(rq);
            b
        })
    }

    /// Prefill: `x_batch[0..n] += bf16(partial[0..n])` from the primary's local
    /// copy of the partial, skipped after any timed-out wait (`error`).
    pub fn hetero_add_partial(&mut self, error: u64, partial: &GpuTensor, n: i32, x_batch: &GpuTensor) -> HipResult<()> {
        let er = error as *mut c_void;
        let (pa, xb) = (partial.buf.as_ptr(), x_batch.buf.as_ptr());
        let mut params = vec![p(&er), p(&pa), p(&n), p(&xb)];
        let grid = (n.max(1) as u32).div_ceil(256).clamp(1, 1024);
        self.hetero_launch("hetero_add_partial", grid, 256, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(er);
            b.push_ptr(pa);
            b.push_i32(n);
            b.push_ptr(xb);
            b
        })
    }

    /// Expert device: `dst[0..n] = f32(src[0..n])`, `src` F16 at a device address.
    pub fn hetero_f16_to_f32(&mut self, src: u64, dst: &GpuTensor, n: i32) -> HipResult<()> {
        let (sr, d) = (src as *mut c_void, dst.buf.as_ptr());
        let mut params = vec![p(&sr), p(&d), p(&n)];
        let grid = (n.max(1) as u32).div_ceil(256).clamp(1, 1024);
        self.hetero_launch("hetero_f16_to_f32", grid, 256, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sr);
            b.push_ptr(d);
            b.push_i32(n);
            b
        })
    }

    /// Expert device: `dst[0..n] = bf16_rne(src[0..n])`, `dst` at a device address.
    pub fn hetero_f32_to_bf16(&mut self, src: &GpuTensor, dst: u64, n: i32) -> HipResult<()> {
        let (sr, d) = (src.buf.as_ptr(), dst as *mut c_void);
        let mut params = vec![p(&sr), p(&d), p(&n)];
        let grid = (n.max(1) as u32).div_ceil(256).clamp(1, 1024);
        self.hetero_launch("hetero_f32_to_bf16", grid, 256, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sr);
            b.push_ptr(d);
            b.push_i32(n);
            b
        })
    }
}
