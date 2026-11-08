#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate the Route-A PeaceMaker twins of `lane_leviathan_verify` (gfx1200, gfx1201).

Explicit invocation only: no test, build script or runtime path runs this. It

  1. compiles `kernels/src/lane_leviathan.hip` with the runtime's `core_hipcc_flags`
     (`--genco --offload-arch=<arch> -O3 --no-offload-compress -fuse-cuid=none
     -mcode-object-version=6`, `rdna-compute/src/compiler.rs`) into the anchor
     `kernels/lane_leviathan_hipcc_<arch>.co`;
  2. seals it with `peacemaker-isa`'s `hip_twin` example (lift, `emit::checked` byte identity,
     bundle, sha-pin) into `kernels/lane_leviathan_pm_<arch>.hxaco` (+ `.hxaco.sha256`);
  3. writes `kernels/lane_leviathan_pm_<arch>.twin.json` (source sha256, compiler provenance,
     portable flags, seal report including the kernarg ABI, optional math evidence).

Tests (`crates/peacemaker-isa/tests/hip_twin_leviathan.rs`) only consume the committed files.

    scripts/pm-leviathan-twin.py [--family leviathan|moe-shared-down-combine]
        [--arch gfx1200|gfx1201|all] [--rocm-path DIR]
        [--build-wrapper 'WRAPPER ARGS'] [--seal-bin PATH] [--out-dir DIR]

`--family moe-shared-down-combine` (gfx1201 only) compiles `kernels/src/moe_shared_down_combine.gfx1201.hip`
once and seals its eight `moe_shared_down_combine_rows_r{1..8}_zinit` symbols into one
`kernels/moe_shared_down_combine_pm_gfx1201.hxaco` (anchor `moe_shared_down_combine_hipcc_gfx1201.co`,
manifest `moe_shared_down_combine_pm_gfx1201.twin.json`, `seal.reports` in symbol order); no math evidence.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ARCHES = ("gfx1200", "gfx1201")

FAMILIES = {
    "leviathan": {
        "source": "kernels/src/lane_leviathan.hip",
        "symbols": ("lane_leviathan_verify",),
        "prefix": "lane_leviathan",
        "arches": ARCHES,
        "math": True,
    },
    "moe-shared-down-combine": {
        "source": "kernels/src/moe_shared_down_combine.gfx1201.hip",
        "symbols": tuple(f"moe_shared_down_combine_rows_r{r}_zinit" for r in range(1, 9)),
        "prefix": "moe_shared_down_combine",
        "arches": ("gfx1201",),
        "math": False,
    },
}


def core_hipcc_flags(arch):
    """Mirror of `rdna-compute` `core_hipcc_flags`."""
    return [
        "--genco",
        f"--offload-arch={arch}",
        "-O3",
        "--no-offload-compress",
        "-fuse-cuid=none",
        "-mcode-object-version=6",
    ]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def die(msg):
    print(f"pm-leviathan-twin: {msg}", file=sys.stderr)
    sys.exit(1)


def find_rocm(arg):
    cands = [arg, os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"), "/opt/rocm"]
    for c in cands:
        if c and (Path(c) / "bin" / "hipcc").is_file():
            return Path(c)
    hipcc = shutil.which("hipcc")
    if hipcc:
        return Path(hipcc).resolve().parent.parent
    die("no ROCm found: pass --rocm-path or set ROCM_PATH (needs bin/hipcc)")


def run(cmd, cwd=None, env=None, capture=True):
    p = subprocess.run(
        cmd, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if p.returncode != 0:
        die(f"command failed ({p.returncode}): {shlex.join(cmd)}\n{p.stderr or ''}")
    return p.stdout


_ABS_PATH = re.compile(r"(^|\s)/[\w.+-]+/")


def compiler_version(hipcc, env):
    """`hipcc --version` minus every line that names a host path (InstalledDir, config file)."""
    out = run([str(hipcc), "--version"], env=env)
    keep = []
    for line in out.splitlines():
        if line.startswith(("InstalledDir", "Configuration file")) or _ABS_PATH.search(line):
            continue
        if line.strip():
            keep.append(line.rstrip())
    return keep


WRAPPERS = {"expf": "_ZL4expff", "powf": "_ZL4powfff"}


def find_llvm_dis(rocm):
    for c in (rocm / "lib" / "llvm" / "bin" / "llvm-dis", rocm / "llvm" / "bin" / "llvm-dis",
              rocm / "bin" / "llvm-dis"):
        if c.is_file():
            return c
    found = shutil.which("llvm-dis")
    return Path(found) if found else None


def math_evidence(hipcc, env, root, arch, tmp, rocm):
    """Where do the source's expf/powf go under the actual product flags?

    Recompiles with the exact core flags plus `-save-temps=obj`, disassembles the pre-link device
    bitcode (`*.tmp.bc`) and records, for each libm wrapper function, the
    callees found in its body plus the extracted wrapper text. Nothing is assumed: the callee
    (e.g. `llvm.exp.f32` vs `__ocml_exp_f32`) is whatever the IR says. The sealed bytes are the
    real anchor regardless of which path a function takes.
    """
    d = Path(tmp) / f"math_{arch}"
    d.mkdir()
    ev = {"method": "core_hipcc_flags + -save-temps=obj; llvm-dis of the pre-link *.tmp.bc"}
    ev["note_scope"] = ("callees are read per wrapper body; an OCML function reached by a wrapper "
                        "may itself call further intrinsics (e.g. __ocml_exp_f32 -> llvm.exp.f32), "
                        "which this record does not follow")
    cmd = [str(hipcc), *core_hipcc_flags(arch), "-save-temps=obj",
           "-o", str(d / "m.co"), SOURCE_REL]
    p = subprocess.run(cmd, cwd=root, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dis = find_llvm_dis(rocm)
    bcs = sorted(d.glob(f"*{arch}*.tmp.bc"))
    if p.returncode != 0 or not bcs or dis is None:
        die("could not prove math paths before sealing (compile rc "
            f"{p.returncode}, tmp.bc files {len(bcs)}, llvm-dis "
            f"{'found' if dis else 'missing'}): {p.stderr}")
    text = subprocess.run([str(dis), str(bcs[0]), "-o", "-"], text=True,
                          stdout=subprocess.PIPE, check=True).stdout
    ev["ir_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    funcs = {}
    for name, mangled in WRAPPERS.items():
        m = re.search(rf"^define [^\n]*@{re.escape(mangled)}\(.*?^}}", text, re.S | re.M)
        if not m:
            die(f"{arch}: no {name} wrapper in saved device IR; math path not proved")
        body = m.group(0)
        callees = sorted(set(re.findall(r"\bcall\b[^\n@]*@([\w.$]+)\(", body)))
        funcs[name] = {"wrapper": mangled, "found": True, "callees": callees,
                       "extract": body.splitlines()}
    ev["functions"] = funcs
    ev["status"] = "recorded"
    return ev


def seal_cmd(args, root):
    if args.seal_bin:
        return [str(Path(args.seal_bin).resolve())]
    wrapper = shlex.split(args.build_wrapper) if args.build_wrapper else []
    return [*wrapper, "cargo", "run", "--release", "-p", "peacemaker-isa",
            "--features", "lab", "--example", "hip_twin", "--"]


def build_arch(args, root, hipcc, env, arch, tmp):
    fam = FAMILIES[args.family]
    source_rel = fam["source"]
    symbols = fam["symbols"]
    prefix = fam["prefix"]
    out_dir = args.out_dir
    co = out_dir / f"{prefix}_hipcc_{arch}.co"
    hxaco = out_dir / f"{prefix}_pm_{arch}.hxaco"
    manifest = out_dir / f"{prefix}_pm_{arch}.twin.json"
    flags = core_hipcc_flags(arch)
    evidence = math_evidence(hipcc, env, root, arch, tmp, Path(env["ROCM_PATH"])) if fam["math"] else None

    tmp_co = Path(tmp) / f"anchor_{arch}.co"
    run([str(hipcc), *flags, "-o", str(tmp_co), source_rel], cwd=root, env=env)
    shutil.copyfile(tmp_co, co)

    report_path = Path(tmp) / f"report_{arch}.json"
    kernel_args = [a for s in symbols for a in ("--kernel", s)]
    run([*seal_cmd(args, root), "--in", str(co), "--arch", arch, "--out", str(hxaco),
         "--report", str(report_path), *kernel_args], cwd=root, env=env)
    report = json.loads(report_path.read_text())

    rel = lambda p: p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name
    doc = {
        "module": f"{prefix}_pm_{arch}",
        "arch": arch,
        "source": {
            "path": source_rel,
            "bytes": (root / source_rel).stat().st_size,
            "sha256": sha256_file(root / source_rel),
        },
        "compiler": {
            "version": compiler_version(hipcc, env),
            "flags": flags,
            "invocation": "hipcc <flags> -o <anchor.co> " + source_rel,
        },
        "anchor": {"file": co.name, "bytes": co.stat().st_size, "sha256": sha256_file(co)},
        "hxaco": {"file": hxaco.name, "bytes": hxaco.stat().st_size, "sha256": sha256_file(hxaco)},
    }
    tool = "peacemaker-isa example hip_twin (Route A: lift, emit::checked, bundle)"
    if len(symbols) == 1:
        doc["symbol"] = symbols[0]
        doc["seal"] = {"tool": tool, "report": report}
    else:
        if not isinstance(report, list) or len(report) != len(symbols):
            die(f"expected {len(symbols)} seal reports, got {report!r:.80}")
        doc["symbols"] = list(symbols)
        doc["seal"] = {"tool": tool, "reports": report}
    if evidence is not None:
        doc["math_evidence"] = evidence
    # Supplemental receipts (m7 OPEN diagnostic, gpu_oracle) are explicit claims about one
    # exact hxaco. Carry them over only when the existing manifest is for the same hxaco and
    # each receipt's own input hash matches; otherwise omit so the CPU audit fails closed.
    new_sha = doc["hxaco"]["sha256"]
    if manifest.exists():
        try:
            old = json.loads(manifest.read_text())
        except ValueError:
            old = {}
        if old.get("hxaco", {}).get("sha256") == new_sha:
            for key in ("m7", "gpu_oracle"):
                rec = old.get(key)
                if isinstance(rec, dict) and rec.get("input_hxaco_sha256") == new_sha:
                    doc[key] = rec
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"{arch}: wrote {rel(co)} {rel(hxaco)} {rel(manifest)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", default="leviathan", choices=tuple(FAMILIES),
                    help="twin family (default: leviathan)")
    ap.add_argument("--arch", default=None, choices=(*ARCHES, "all"),
                    help="default: all for leviathan, gfx1201 for moe-shared-down-combine")
    ap.add_argument("--rocm-path", help="ROCm root (default: $ROCM_PATH, $HIP_PATH, /opt/rocm, PATH)")
    ap.add_argument("--build-wrapper", default=None,
                    help="command prefix for the cargo invocation, e.g. a build-slot script")
    ap.add_argument("--seal-bin", help="prebuilt hip_twin example binary (skips cargo)")
    ap.add_argument("--out-dir", type=Path, help="artifact directory (default: <repo>/kernels)")
    args = ap.parse_args()
    fam = FAMILIES[args.family]

    root = Path(__file__).resolve().parent.parent
    if not (root / fam["source"]).is_file():
        die(f"{fam['source']} not found under {root}")
    args.out_dir = (args.out_dir or root / "kernels").resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rocm = find_rocm(args.rocm_path)
    hipcc = rocm / "bin" / "hipcc"
    env = dict(os.environ, ROCM_PATH=str(rocm))

    allowed = fam["arches"]
    arch_arg = args.arch or ("all" if len(allowed) > 1 else allowed[0])
    arches = allowed if arch_arg == "all" else (arch_arg,)
    if any(a not in allowed for a in arches):
        die(f"family {args.family} supports only {', '.join(allowed)}")
    work = root / "target" / "pm-leviathan-twin"
    work.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(work)
    with tempfile.TemporaryDirectory(prefix="run-", dir=work) as tmp:
        for arch in arches:
            build_arch(args, root, hipcc, env, arch, tmp)


if __name__ == "__main__":
    main()
