#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Convert a rocprofv3 rocpd sqlite trace (`*_results.db`) into the CSV set
`pm-census` reads (crates/pm-census/src/rocprof.rs `load_csv`):

    <outdir>/<prefix>kernel_trace.csv   kernel dispatches
    <outdir>/<prefix>agent_info.csv     agents
    <outdir>/<prefix>memory_copy_trace.csv  (only when the db has copies)

Usage: rocpd_to_csv.py <db> <outdir> [--prefix PREFIX]

Python 3 stdlib only. The db is opened read-only. Sources are the rocpd
`kernels`, `memory_copies` and `rocpd_info_agent` views (names are the views',
not the per-guid tables, so no guid suffix is needed).

Agent ids: rocprofv3 CSVs name an agent "Agent N" where N is the agent's
logical node id, and `agent_info.csv` carries `Logical_Node_Id`; the loader
joins the two. The rocpd views expose both `agent_abs_index` and
`agent_log_index`; this script writes the *logical* index in both places so
the join holds by construction (they are equal on single-process traces).
"""
import argparse
import csv
import os
import sqlite3
import sys

KERNEL_COLS = [
    "Kernel_Name", "Kind", "Correlation_Id", "Agent_Id", "Queue_Id", "Stream_Id",
    "Thread_Id", "Dispatch_Id", "Kernel_Id", "Start_Timestamp", "End_Timestamp",
    "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count",
    "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
    "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
]
COPY_COLS = [
    "Kind", "Direction", "Stream_Id", "Source_Agent_Id", "Destination_Agent_Id",
    "Correlation_Id", "Start_Timestamp", "End_Timestamp", "Bytes",
]
AGENT_COLS = ["Node_Id", "Logical_Node_Id", "Agent_Type", "Name"]


def table_columns(con, view):
    return [r[1] for r in con.execute(f"PRAGMA table_info({view})")]


def require(con, view, needed):
    have = set(table_columns(con, view))
    if not have:
        sys.exit(f"rocpd_to_csv: db has no {view!r} view")
    missing = [c for c in needed if c not in have]
    if missing:
        sys.exit(f"rocpd_to_csv: view {view!r} lacks columns {missing}")
    return have


def n(v):
    return 0 if v is None else v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("db")
    ap.add_argument("outdir")
    ap.add_argument("--prefix", default=None,
                    help="file-name prefix (default: the db file stem + '_')")
    args = ap.parse_args()

    if not os.path.isfile(args.db):
        sys.exit(f"rocpd_to_csv: {args.db}: no such file")
    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.prefix
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(args.db))[0] + "_"

    con = sqlite3.connect(f"file:{os.path.abspath(args.db)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    kcols = require(con, "kernels", [
        "name", "start", "end", "grid_x", "grid_y", "grid_z", "workgroup_x",
        "workgroup_y", "workgroup_z", "lds_size", "scratch_size", "vgpr_count",
        "sgpr_count", "queue_id", "kernel_id", "corr_id",
    ])
    agent_col = "agent_log_index" if "agent_log_index" in kcols else "agent_abs_index"
    if agent_col not in kcols:
        sys.exit("rocpd_to_csv: view 'kernels' has neither agent_log_index nor agent_abs_index")
    opt = lambda c, default="0": c if c in kcols else default  # noqa: E731

    kernel_path = os.path.join(args.outdir, f"{prefix}kernel_trace.csv")
    n_kernels = 0
    # ORDER BY start then dispatch id: rocprofv3's own csv is dispatch-ordered and
    # pm-census derives pass/chunk structure from that order.
    order = "start, " + (opt("dispatch_id", "id"))
    query = f"""
        SELECT name, {agent_col} AS agent, queue_id, kernel_id, corr_id, start, "end",
               lds_size, scratch_size, vgpr_count, {opt('accum_vgpr_count')} AS accum_vgpr,
               sgpr_count, workgroup_x, workgroup_y, workgroup_z, grid_x, grid_y, grid_z,
               {opt('stream_id')} AS stream_id, {opt('tid')} AS tid,
               {opt('dispatch_id', 'id')} AS dispatch_id
        FROM kernels ORDER BY {order}
    """
    with open(kernel_path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
        w.writerow(KERNEL_COLS)
        for r in con.execute(query):
            w.writerow([
                r["name"], "KERNEL_DISPATCH", n(r["corr_id"]), f"Agent {r['agent']}",
                n(r["queue_id"]), n(r["stream_id"]), n(r["tid"]), n(r["dispatch_id"]),
                n(r["kernel_id"]), r["start"], r["end"],
                n(r["lds_size"]), n(r["scratch_size"]), n(r["vgpr_count"]),
                n(r["accum_vgpr"]), n(r["sgpr_count"]),
                r["workgroup_x"], r["workgroup_y"], r["workgroup_z"],
                r["grid_x"], r["grid_y"], r["grid_z"],
            ])
            n_kernels += 1

    agent_path = os.path.join(args.outdir, f"{prefix}agent_info.csv")
    acols = require(con, "rocpd_info_agent", ["type", "name"])
    log_col = "logical_index" if "logical_index" in acols else "absolute_index"
    abs_col = "absolute_index" if "absolute_index" in acols else log_col
    with open(agent_path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
        w.writerow(AGENT_COLS)
        for r in con.execute(
            f"SELECT {abs_col} AS a, {log_col} AS l, type, name FROM rocpd_info_agent ORDER BY l"
        ):
            w.writerow([r["a"], r["l"], r["type"], r["name"]])

    n_copies = 0
    ccols = set(table_columns(con, "memory_copies"))
    if {"start", "end", "name"} <= ccols:
        src = "src_agent_log_index" if "src_agent_log_index" in ccols else "src_agent_abs_index"
        dst = "dst_agent_log_index" if "dst_agent_log_index" in ccols else "dst_agent_abs_index"
        # src/dst agent columns are the log indices, matching the kernel Agent_Id
        sel = (f'SELECT name, {src} AS src, {dst} AS dst, start, "end", '
               f'{"size" if "size" in ccols else "0"} AS size, '
               f'{"stream_id" if "stream_id" in ccols else "0"} AS stream_id, '
               f'{"corr_id" if "corr_id" in ccols else "0"} AS corr_id '
               "FROM memory_copies ORDER BY start")
        copy_rows = list(con.execute(sel))
        if copy_rows:
            copy_path = os.path.join(args.outdir, f"{prefix}memory_copy_trace.csv")
            with open(copy_path, "w", newline="") as f:
                w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
                w.writerow(COPY_COLS)
                for r in copy_rows:
                    direction = r["name"] or ""
                    if not direction.startswith("MEMORY_COPY_"):
                        direction = "MEMORY_COPY_" + direction
                    w.writerow([
                        "MEMORY_COPY", direction, n(r["stream_id"]),
                        f"Agent {r['src']}", f"Agent {r['dst']}", n(r["corr_id"]),
                        r["start"], r["end"], n(r["size"]),
                    ])
                    n_copies += 1

    print(f"{kernel_path}: {n_kernels} kernel dispatches")
    print(f"{agent_path}")
    if n_copies:
        print(f"{n_copies} memory copies")


if __name__ == "__main__":
    main()
