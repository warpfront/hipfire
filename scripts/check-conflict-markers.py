#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Fail on unresolved merge-conflict markers in tracked text files.

A cherry-pick/merge resolution that keeps a marker line (`<<<<<<< `, `||||||| ` diff3 base, `>>>>>>> `, or a
`=======` separator inside such a block) is a broken resolution even when everything still builds: the
CHANGELOG is the usual victim. Markers are only recognised at the start of a line, exactly as git writes them;
a bare `=======` counts only between an opening `<<<<<<< ` and a closing `>>>>>>> ` (so markdown setext
underlines are not flagged). Binary files (NUL in the first 8 KiB) are skipped.

usage: check-conflict-markers.py [PATH ...]   (default: every file `git ls-files` lists)
exit 0 = clean, 1 = markers found (each printed as path:line: text).
"""
import subprocess
import sys

OPEN, BASE, CLOSE, SEP = "<<<<<<< ", "||||||| ", ">>>>>>> ", "======="


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def scan(path):
    try:
        data = open(path, "rb").read()
    except (FileNotFoundError, IsADirectoryError):
        return []
    if b"\0" in data[:8192]:
        return []
    hits, inside = [], False
    for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        if line.startswith(OPEN):
            inside = True
            hits.append((n, line))
        elif line.startswith(BASE) or line.startswith(CLOSE):
            hits.append((n, line))
            if line.startswith(CLOSE):
                inside = False
        elif inside and line.rstrip() == SEP:
            hits.append((n, line))
    return hits


def main(argv):
    paths = argv or tracked()
    bad = 0
    for p in paths:
        for n, line in scan(p):
            print(f"{p}:{n}: {line[:120]}")
            bad += 1
    if bad:
        print(f"check-conflict-markers: {bad} unresolved conflict-marker line(s)")
        return 1
    print(f"check-conflict-markers: {len(paths)} tracked file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
