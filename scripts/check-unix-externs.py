#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Fail on POSIX/Linux-only libc symbols declared in an `extern` block that Windows compiles.

A raw `extern "C" { fn clock_gettime(..); }` compiles on every target; on x86_64-pc-windows-msvc it
only fails at link time (LNK2019/LNK1120), and only if a reachable path calls it. `cargo check` and
`cargo xwin check` never link, so the break reaches users first (#979, #1116, #1161). This gate finds
every such declaration that is not compiled out on Windows.

A declaration is compiled out when it sits under a `cfg` that is false on Windows: on the `extern`
block, on any enclosing item or statement (`#[cfg(unix)] fn f() { .. }`, `#[cfg(..)] ONCE.call_once(..)`),
an inner `#![cfg(..)]` of the file, or a `#[cfg(..)] mod m;` on any module path down to the file
(`#[path]` honoured). `cfg` is evaluated for x86_64-pc-windows-msvc with three-valued logic:
`unix`, `target_os = "linux"`, `not(windows)`, `all(unix, feature = "x")` are false there;
`feature = ..`, `test`, `debug_assertions` and other unknown predicates may be true, so they never
guard on their own.

usage: check-unix-externs.py [--self-test] [PATH ...]   (default: every `git ls-files '*.rs'`)
exit 0 = clean, 1 = unguarded declaration(s) found (each printed as path:line: symbol).
"""
import os
import re
import subprocess
import sys

# libc symbols with no MSVC CRT / oldnames.lib definition. Extend when a new one is declared.
POSIX_ONLY = {
    "clock_gettime", "clock_nanosleep", "nanosleep", "usleep",
    "mmap", "mmap64", "munmap", "mremap", "madvise", "posix_madvise", "mprotect", "msync",
    "mlock", "munlock", "mlockall", "munlockall", "mincore", "posix_memalign",
    "dlopen", "dlsym", "dlclose", "dlerror", "dladdr", "dlinfo", "dl_iterate_phdr",
    "ioctl", "fcntl", "flock", "poll", "ppoll", "select", "pselect",
    "sysconf", "getpagesize", "syscall", "prctl", "gettid", "getuid", "geteuid", "getgid",
    "getrlimit", "setrlimit", "getrusage", "uname", "statvfs", "fstatvfs",
    "kill", "fork", "vfork", "waitpid", "setsid", "sigaction", "sigprocmask", "pipe2",
    "eventfd", "memfd_create", "shm_open", "shm_unlink", "ftruncate", "pread", "pwrite",
    "readlink", "symlink", "realpath", "mkfifo", "socketpair", "sendmsg", "recvmsg",
    "sched_setaffinity", "sched_getaffinity", "sched_getcpu", "sched_yield",
}
POSIX_PREFIXES = ("pthread_", "epoll_", "inotify_", "timerfd_", "io_uring_")

WINDOWS = {"windows": True, "unix": False}
WINDOWS_KV = {
    "target_os": "windows",
    "target_family": "windows",
    "target_env": "msvc",
    "target_vendor": "pc",
    "target_arch": "x86_64",
    "target_pointer_width": "64",
    "target_endian": "little",
}

# Targets that never build for Windows, so their declarations cannot reach a Windows link. Every entry
# says why; an entry whose path no longer exists fails the gate. Do not add product code here: guard
# it with #[cfg(unix)] / #[cfg(target_os = "linux")] instead.
LINUX_ONLY = {
    "crates/railgun/src/npu/": "railgun's `npu` feature; only `[target.'cfg(target_os = \"linux\")'.dependencies]` enable it",
    "crates/npu-tools/": "XDNA2 (amdxdna) NPU binaries; depend on railgun `npu`, which does not compile off Linux",
    "crates/hip-bridge/examples/shared_vmm_fd_probe.rs": "POSIX-FD VMM export probe (std::os::unix, SCM_RIGHTS)",
    "crates/hipfire-dispatch/examples/d2_mapped_return_pilot.rs": "KFD mapped-return timing pilot (rocprofv3 CLOCK_MONOTONIC)",
    "crates/redline-rocr/examples/pm4_scratch_probe.rs": "ROCr/KFD PM4 scratch probe (KFD userptr eviction)",
}


def is_posix_only(name):
    return name in POSIX_ONLY or name.startswith(POSIX_PREFIXES)


# ---------------------------------------------------------------- cfg evaluation (None = unknown)
def cfg_eval(text):
    toks = re.findall(r'"(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*|[(),=]', text)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def pred():
        name = take()
        if peek() == "(":
            take()
            args = []
            while peek() not in (")", None):
                args.append(pred())
                if peek() == ",":
                    take()
            take()
            if name == "all":
                return False if False in args else (True if all(a is True for a in args) else None)
            if name == "any":
                return True if True in args else (False if all(a is False for a in args) else None)
            if name == "not" and len(args) == 1:
                return None if args[0] is None else not args[0]
            return None
        if peek() == "=":
            take()
            value = take().strip('"')
            if name in WINDOWS_KV:
                return WINDOWS_KV[name] == value
            return None
        return WINDOWS.get(name)

    try:
        return pred()
    except IndexError:
        return None


def windows_off(attr):
    """True when `attr` (an attribute body such as `cfg(unix)`) compiles its item out on Windows."""
    m = re.fullmatch(r"\s*cfg\s*\((.*)\)\s*", attr, re.S)
    return bool(m) and cfg_eval(m.group(1)) is False


# ---------------------------------------------------------------- lexer
TOKEN = re.compile(
    r"""(?P<ws>\s+)
      |(?P<lcomment>//[^\n]*)
      |(?P<bcomment>/\*)
      |(?P<rstr>b?r(?P<hashes>\#*)")
      |(?P<str>b?"(?:[^"\\]|\\.)*")
      |(?P<char>b?'(?:[^'\\\n]|\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.))')
      |(?P<lifetime>'[A-Za-z_][A-Za-z0-9_]*)
      |(?P<ident>[A-Za-z_][A-Za-z0-9_]*)
      |(?P<punct>.)""",
    re.X | re.S,
)


def tokens(src):
    """Yield (kind, text, line); comments dropped, string bodies kept as one token."""
    i, line = 0, 1
    n = len(src)
    while i < n:
        m = TOKEN.match(src, i)
        kind = m.lastgroup if m.lastgroup != "hashes" else "rstr"
        if m.group("rstr") is not None:
            kind = "rstr"
        if kind == "bcomment":
            depth, j = 1, m.end()
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            line += src.count("\n", i, j)
            i = j
            continue
        if kind == "rstr":
            close = '"' + m.group("hashes")
            j = src.find(close, m.end())
            j = n if j < 0 else j + len(close)
            text = src[i:j]
            yield "str", text, line
            line += text.count("\n")
            i = j
            continue
        text = m.group(0)
        if kind not in ("ws", "lcomment"):
            yield ("str" if kind == "char" else kind), text, line
        line += text.count("\n")
        i = m.end()


# ---------------------------------------------------------------- per-file scan
ITEM_KEYWORDS = {"fn", "mod", "impl", "trait", "struct", "enum", "union", "extern", "macro_rules"}
ITEM_PREFIXES = {"pub", "unsafe", "async", "const", "default", "crate", "self", "super", "in"}


class Frame:
    def __init__(self, guarded, item=False, extern=False):
        self.guarded = guarded  # compiled out on Windows
        self.item = item  # an item body: a statement guard does not leak out of it
        self.extern = extern  # an `extern "ABI" { .. }` block
        self.stmt_guard = False  # the current statement at this depth is cfg'd out
        self.stmt_item = None  # the current statement is an item (ends at its closing brace)
        self.stmt_started = False
        self.pending = []  # outer attributes waiting for the next statement


def scan(src):
    """Return (inner_off, mods, hits): file-level `#![cfg]` off on Windows; `mod x;` declarations as
    (name, path_attr, guarded); extern declarations of POSIX-only symbols as (line, name, guarded)."""
    toks = list(tokens(src))
    stack = [Frame(False, item=True)]
    inner_off = False
    mods, hits = [], []
    i = 0

    def attr_body(k):
        depth, j = 0, k
        while j < len(toks):
            if toks[j][1] == "[":
                depth += 1
            elif toks[j][1] == "]":
                depth -= 1
                if depth == 0:
                    return " ".join(t[1] for t in toks[k + 1 : j]), j + 1
            j += 1
        return "", len(toks)

    def guarded(frame):
        return frame.guarded or frame.stmt_guard

    def end_stmt(frame):
        frame.stmt_guard, frame.stmt_item, frame.stmt_started = False, None, False
        frame.pending = []

    while i < len(toks):
        kind, text, line = toks[i]
        top = stack[-1]
        if text == "#" and i + 1 < len(toks) and toks[i + 1][1] in ("[", "!"):
            inner = toks[i + 1][1] == "!"
            body, i = attr_body(i + 2 if inner else i + 1)
            if inner:
                if windows_off(body):
                    if len(stack) == 1:
                        inner_off = True
                    top.guarded = True
            else:
                top.pending.append(body)
            continue
        if not top.stmt_started:
            top.stmt_started = True
            top.stmt_guard = any(windows_off(a) for a in top.pending)
            top.pending_path = next(
                (re.sub(r'^path\s*=\s*"(.*)"$', r"\1", a.strip()) for a in top.pending if a.strip().startswith("path ")),
                None,
            )
            j = i
            while j < len(toks) and (toks[j][1] in ITEM_PREFIXES or toks[j][0] == "str" or toks[j][1] in "()"):
                j += 1
            top.stmt_item = toks[j][1] if j < len(toks) and toks[j][1] in ITEM_KEYWORDS else None
        if text == "{":
            is_extern = False
            k = i - 1
            if k >= 0 and toks[k][0] == "str" and k >= 1 and toks[k - 1][1] == "extern":
                is_extern = True
            elif k >= 0 and toks[k][1] == "extern":
                is_extern = True
            stack.append(Frame(guarded(top), item=top.stmt_item is not None, extern=is_extern))
            i += 1
            continue
        if text == "}":
            if len(stack) > 1:
                stack.pop()
            parent = stack[-1]
            if parent.stmt_item is not None:
                end_stmt(parent)
            i += 1
            continue
        if text == ";":
            if top.stmt_item == "mod" and toks[i - 2][1] == "mod":
                mods.append((toks[i - 1][1], top.pending_path, guarded(top)))
            end_stmt(top)
            i += 1
            continue
        if text == "," and not top.item and not top.extern:
            # struct-literal / match-arm / argument attributes end at the comma
            if top.stmt_item is None and top.stmt_guard:
                end_stmt(top)
            i += 1
            continue
        if top.extern and text == "fn" and i + 1 < len(toks) and toks[i + 1][0] == "ident":
            name = toks[i + 1][1]
            if is_posix_only(name):
                hits.append((toks[i + 1][2], name, guarded(top)))
        i += 1
    return inner_off, mods, hits


# ---------------------------------------------------------------- module tree
def child_paths(parent, name, path_attr):
    d = os.path.dirname(parent)
    if path_attr:
        return [os.path.normpath(os.path.join(d, path_attr))]
    base = os.path.basename(parent)
    if base not in ("lib.rs", "main.rs", "mod.rs") and not parent.endswith("/build.rs"):
        d = os.path.join(d, base[:-3])
    return [os.path.normpath(os.path.join(d, name + ".rs")), os.path.normpath(os.path.join(d, name, "mod.rs"))]


RELEVANT = re.compile(r'\bextern\s+"|\bmod\s+\w+\s*;')


def check(files, read=lambda p: open(p, encoding="utf-8", errors="replace").read(), linux_only=LINUX_ONLY):
    files = [p for p in files if not any(p.startswith(prefix) for prefix in linux_only)]
    # A file with neither an extern block nor a `mod x;` declaration has no hits and no children.
    texts = {p: read(p) for p in files}
    scans = {p: scan(t) if RELEVANT.search(t) else (False, [], []) for p, t in texts.items()}
    # file -> list of (parent, guarded decl)
    parents = {}
    for p, (_, mods, _) in scans.items():
        for name, path_attr, g in mods:
            for c in child_paths(p, name, path_attr):
                if c in scans:
                    parents.setdefault(c, []).append((p, g))
                    break
    memo = {}

    def off(p, seen=()):
        if p in memo:
            return memo[p]
        if p in seen:
            return False
        result = scans[p][0]
        if not result and p in parents:
            # compiled out only if every declaration that pulls it in is compiled out
            result = all(g or off(par, seen + (p,)) for par, g in parents[p])
        memo[p] = result
        return result

    bad = []
    for p in sorted(scans):
        if off(p):
            continue
        for line, name, g in scans[p][2]:
            if not g:
                bad.append((p, line, name))
    return bad


SELF_TEST = {
    "a/src/lib.rs": """
        #[cfg(unix)] mod posix;
        mod open;
        #[path = "plat/lin.rs"] #[cfg(target_os = "linux")] mod lin;
        #[cfg(target_os = "linux")]
        fn guarded_fn() { extern "C" { fn clock_gettime(c: i32, t: *mut u8) -> i32; } }
        fn bad_fn() -> u64 { extern "C" { fn clock_gettime(c: i32, t: *mut u8) -> i32; } 0 }
        fn stmt() {
            #[cfg(not(target_os = "linux"))]
            ONCE.call_once(|| eprintln!("x"));
            #[cfg(target_os = "linux")]
            ONCE.call_once(move || { unsafe extern "C" { fn madvise(a: *mut u8, l: usize, x: i32) -> i32; } });
            ONCE.call_once(move || { unsafe extern "C" { fn mmap(a: *mut u8) -> *mut u8; } });
        }
        #[cfg(all(unix, feature = "x"))] extern "C" { fn dlopen(n: *const u8, f: i32) -> *mut u8; }
        #[cfg(feature = "x")] extern "C" { fn dlsym(h: *mut u8, n: *const u8) -> *mut u8; }
        #[cfg(windows)] extern "system" { fn GetTickCount() -> u32; }
        extern "C" { fn hipInit(f: u32) -> u32; }
        fn mmap() {}
        const S: &str = "extern \\"C\\" { fn ioctl(); }";
        // extern "C" { fn ioctl(); }
    """,
    "a/src/posix.rs": 'extern "C" { fn ioctl(fd: i32, r: u64, ...) -> i32; }',
    "a/src/open.rs": '#![cfg(unix)]\nextern "C" { fn pthread_self() -> u64; }',
    "a/src/plat/lin.rs": 'extern "C" { fn sysconf(n: i32) -> i64; }',
    "a/src/bin/tool.rs": 'extern "C" { fn getrlimit(r: i32, l: *mut u64) -> i32; }',
    "b/src/lib.rs": 'extern "C" { fn ioctl(fd: i32, r: u64, ...) -> i32; }',
}


def self_test():
    got = sorted((p, n) for p, _, n in check(sorted(SELF_TEST), read=SELF_TEST.__getitem__, linux_only={"b/": "test"}))
    want = sorted(
        [("a/src/lib.rs", "clock_gettime"), ("a/src/lib.rs", "mmap"), ("a/src/lib.rs", "dlsym"), ("a/src/bin/tool.rs", "getrlimit")]
    )
    if got != want:
        print(f"check-unix-externs --self-test: FAILED\n  want {want}\n  got  {got}")
        return 1
    print(f"check-unix-externs --self-test: ok ({len(want)} planted defects found, guarded ones ignored)")
    return 0


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    stale = [] if argv else [prefix for prefix in LINUX_ONLY if not os.path.exists(prefix)]
    files = argv or [
        p for p in subprocess.run(["git", "ls-files", "-z", "*.rs"], capture_output=True, check=True).stdout.decode().split("\0") if p
    ]
    bad = check([os.path.normpath(p) for p in files])
    for p, line, name in bad:
        print(f"{p}:{line}: `{name}` declared in an extern block that Windows compiles (wrap it in #[cfg(unix)])")
    for prefix in stale:
        print(f"check-unix-externs: LINUX_ONLY entry {prefix!r} no longer exists; delete it")
    if bad or stale:
        print(f"check-unix-externs: {len(bad)} POSIX-only extern declaration(s) not compiled out on Windows")
        return 1
    print(f"check-unix-externs: {len(files)} Rust file(s) clean ({len(LINUX_ONLY)} Linux-only target(s) skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
