#!/usr/bin/env -S python3 -I
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Trace-fixture rows of a walked route plan.

`crates/rdna-compute/tests/fixtures/kernel-trace-*.tsv` rows are `arch<TAB>module<TAB>symbols`: the HIP modules a
run's cold kernel cache wrote back, with the symbols it resolved. A route walk (`route_plan_walk.py` +
`route_plan_build.py`) records the same facts, so the rows for an arch whose family has no cold-cache trace are the
union of the plan's HIP rows over both spec routes (`native_mtp` rows include every `ar` row). Embedded images are not
cold-cache modules and are not trace rows.

usage: route_plan_trace_rows.py PLAN.tsv > rows.tsv      (prints a comment header, then the rows, sorted by module)
"""
import pathlib
import sys


def trace_rows(plan_text):
    """(arch, [(module, [symbols])]) of a route plan's HIP rows, symbols unioned across spec routes."""
    arch, modules = None, {}
    for line in plan_text.splitlines():
        if line.startswith("#! arch "):
            arch = line.split(" ", 2)[2]
        if line.startswith("#") or not line:
            continue
        route, kind, module, symbols, _sha = line.split("\t")
        if route not in ("ar", "native_mtp") or kind not in ("hip", "embedded"):
            raise SystemExit(f"malformed plan row {line!r}")
        if kind == "hip":
            modules.setdefault(module, set()).update(symbols.split(","))
    if arch is None:
        raise SystemExit("plan has no #! arch directive")
    return arch, [(module, sorted(symbols)) for module, symbols in sorted(modules.items())]


def render(plan_name, plan_text):
    arch, rows = trace_rows(plan_text)
    out = [f"# {arch}: the HIP rows of route_plans/{plan_name} (module-load route walk, scripts/route_plan_walk.py"
           " --profile xtx-fn-bounded),", "# union of the ar and native_mtp routes; embedded images are not cold-cache modules."]
    out += [f"{arch}\t{module}\t{','.join(symbols)}" for module, symbols in rows]
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = pathlib.Path(sys.argv[1])
    sys.stdout.write(render(path.name, path.read_text()))


if __name__ == "__main__":
    main()
