// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The recorded real route (`layerN.trace.json` + the `route.bin` it names) and the NT1 tile construction of the R6 Gp
//! driver (`qwen4_gp_k2_timing`): same chunks, same padded 16-slot tiles, same slot ids `token * topk + j`.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::common::*;
use super::records::EXPERTS;

pub const TRACE_SCHEMA: &str = "subq4.gp.route-trace.v1";
pub const GRID_Y_MAX: usize = 65535;

pub struct Chunk {
    pub rows: usize,
    pub start_pos: usize,
    /// `rows * topk` expert ids, `[row][j]`.
    pub ids: Vec<i32>,
}

pub struct Route {
    pub layer: usize,
    pub token_count: usize,
    pub topk: usize,
    pub rows_per_expert: Vec<usize>,
    pub chunks: Vec<Chunk>,
    pub trace_path: PathBuf,
    pub trace_raw_sha256: String,
    pub trace_json: Value,
    pub route_path: PathBuf,
    pub route_sha256: String,
    pub route_bytes: usize,
}

pub fn load_route(trace_path: &Path, route_override: Option<&Path>, layer: usize) -> Result<Route> {
    let (tj, traw) = read_json(trace_path)?;
    if tj["schema"].as_str() != Some(TRACE_SCHEMA) {
        return Err(format!(
            "{}: schema {:?} != {TRACE_SCHEMA}",
            trace_path.display(),
            tj["schema"]
        ));
    }
    if tj["layer"].as_u64() != Some(layer as u64) {
        return Err(format!(
            "{}: layer {:?} != {layer}",
            trace_path.display(),
            tj["layer"]
        ));
    }
    let token_count = json_u64(&tj["token_count"], "token_count")? as usize;
    let topk = json_u64(&tj["top_k"], "top_k")? as usize;
    let experts = json_u64(&tj["experts"], "experts")? as usize;
    if experts != EXPERTS {
        return Err(format!("trace experts {experts} != {EXPERTS}"));
    }
    let rpe: Vec<usize> = tj["rows_per_expert"]
        .as_array()
        .ok_or("rows_per_expert")?
        .iter()
        .map(|x| json_u64(x, "rows_per_expert[]").map(|u| u as usize))
        .collect::<Result<_>>()?;
    if rpe.len() != experts || rpe.iter().sum::<usize>() != token_count * topk {
        return Err(format!(
            "trace rows_per_expert: len {} sum {} != {experts} / {}",
            rpe.len(),
            rpe.iter().sum::<usize>(),
            token_count * topk
        ));
    }
    let route_path = match route_override {
        Some(p) => p.to_path_buf(),
        None => PathBuf::from(json_str(&tj["input"]["path"], "input.path")?),
    };
    let raw = read_file(&route_path)?;
    let route_sha256 = sha(&raw);
    if Some(route_sha256.as_str()) != tj["input"]["sha256"].as_str()
        || Some(raw.len() as u64) != tj["input"]["bytes"].as_u64()
    {
        return Err(format!(
            "{}: sha256 {route_sha256} / {} bytes != trace input {:?}",
            route_path.display(),
            raw.len(),
            tj["input"]
        ));
    }
    // route.bin: repeated [4 x u32le (layer, rows, topk, start_pos)] + rows*topk x i32le ids, all layers interleaved.
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut pos = 0usize;
    while pos < raw.len() {
        let hdr = raw
            .get(pos..pos + 16)
            .ok_or("route.bin: truncated record header")?;
        let w = |i: usize| u32::from_le_bytes(hdr[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
        let (l, rows, tk, start) = (w(0), w(1), w(2), w(3));
        let n = rows * tk;
        let body = raw
            .get(pos + 16..pos + 16 + n * 4)
            .ok_or("route.bin: truncated record body")?;
        pos += 16 + n * 4;
        if l != layer {
            continue;
        }
        if tk != topk {
            return Err(format!(
                "route.bin: layer {layer} record topk {tk} != {topk}"
            ));
        }
        let ids: Vec<i32> = body
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if ids.iter().any(|&e| e < 0 || e as usize >= experts) {
            return Err(format!(
                "route.bin: layer {layer} expert id outside [0, {experts})"
            ));
        }
        chunks.push(Chunk {
            rows,
            start_pos: start,
            ids,
        });
    }
    chunks.sort_by_key(|c| c.start_pos);
    let mut next = 0usize;
    for c in &chunks {
        if c.start_pos != next {
            return Err(format!("route.bin: layer {layer} chunks do not tile [0, {token_count}) (gap/overlap at {next})"));
        }
        next += c.rows;
    }
    if next != token_count {
        return Err(format!(
            "route.bin: layer {layer} chunks cover {next} tokens, trace says {token_count}"
        ));
    }
    let mut counts = vec![0usize; experts];
    for c in &chunks {
        for &e in &c.ids {
            counts[e as usize] += 1;
        }
    }
    if counts != rpe {
        return Err(format!(
            "route.bin: layer {layer} rows_per_expert recomputed from the capture differs from {}",
            trace_path.display()
        ));
    }
    let want_chunks: Vec<(u64, u64)> = tj["chunks"]
        .as_array()
        .ok_or("chunks")?
        .iter()
        .map(|c| {
            (
                c["rows"].as_u64().unwrap_or(u64::MAX),
                c["start_pos"].as_u64().unwrap_or(u64::MAX),
            )
        })
        .collect();
    let got_chunks: Vec<(u64, u64)> = chunks
        .iter()
        .map(|c| (c.rows as u64, c.start_pos as u64))
        .collect();
    if want_chunks != got_chunks {
        return Err(format!(
            "trace chunks {want_chunks:?} != route.bin chunks {got_chunks:?}"
        ));
    }
    Ok(Route {
        layer,
        token_count,
        topk,
        rows_per_expert: rpe,
        chunks,
        trace_path: trace_path.to_path_buf(),
        trace_raw_sha256: sha(&traw),
        trace_json: tj,
        route_path,
        route_sha256,
        route_bytes: raw.len(),
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tiling {
    Aggregate,
    Chunked,
}
impl Tiling {
    pub fn name(self) -> &'static str {
        match self {
            Tiling::Aggregate => "aggregate",
            Tiling::Chunked => "chunked",
        }
    }
}

/// One launch pair's worth of tiles (a whole trace for `aggregate`, one capture chunk for `chunked`).
pub struct TileSet {
    pub chunk: Option<usize>,
    /// x rows of the gate/up activations (tokens).
    pub tokens: usize,
    pub topk: usize,
    pub tile_expert: Vec<i32>,
    pub sorted: Vec<i32>,
    pub live_slots: usize,
}

impl TileSet {
    pub fn ntiles(&self) -> usize {
        self.tile_expert.len()
    }
}

/// NT1 padded 16-slot tiles per pool expert. `pool_pos[e]` is the pool position of expert `e` (or -1 = outside scope).
pub fn build_tiles(
    route: &Route,
    pool_pos: &[i32],
    npool: usize,
    tiling: Tiling,
) -> Result<Vec<TileSet>> {
    let topk = route.topk;
    let collect = |chunks: &[&Chunk],
                   base_of: &dyn Fn(&Chunk) -> usize,
                   tokens: usize,
                   chunk: Option<usize>|
     -> Result<TileSet> {
        let mut per: Vec<Vec<i32>> = vec![Vec::new(); npool];
        for c in chunks {
            let base = base_of(c);
            for (idx, &e) in c.ids.iter().enumerate() {
                let p = pool_pos[e as usize];
                if p >= 0 {
                    per[p as usize].push((base + idx) as i32);
                }
            }
        }
        let (mut tile_expert, mut sorted, mut live) = (Vec::new(), Vec::new(), 0usize);
        for (p, slots) in per.iter().enumerate() {
            for t in slots.chunks(TILE_ROWS) {
                tile_expert.push(p as i32);
                sorted.extend_from_slice(t);
                sorted.extend(std::iter::repeat_n(-1i32, TILE_ROWS - t.len()));
                live += t.len();
            }
        }
        if tile_expert.len() > GRID_Y_MAX {
            return Err(format!(
                "{} tiles exceed the grid.y limit {GRID_Y_MAX}",
                tile_expert.len()
            ));
        }
        Ok(TileSet {
            chunk,
            tokens,
            topk,
            tile_expert,
            sorted,
            live_slots: live,
        })
    };
    match tiling {
        Tiling::Aggregate => {
            let all: Vec<&Chunk> = route.chunks.iter().collect();
            Ok(vec![collect(
                &all,
                &|c: &Chunk| c.start_pos * topk,
                route.token_count,
                None,
            )?])
        }
        Tiling::Chunked => route
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| collect(&[c], &|_: &Chunk| 0, c.rows, Some(i)))
            .collect(),
    }
}
