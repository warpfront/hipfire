#!/usr/bin/env python3
"""Developer-only, sampled VRAM guard; NOT a hard physical allocation limit.

Example (inherits the existing managed-allocation emulator unchanged):
  HIPFIRE_EMU=rx7900xt python3 scripts/tiered_kv_vram_guard.py \
      --pci-bdf 0000:03:00.0 --out run.vram.json -- COMMAND ARG ...

The emulator's 20 GiB managed cap and this device-wide sysfs observation are
complementary: driver/JIT/other allocations also appear in the latter. Short
spikes between 50 ms samples can be missed, and allocations can overshoot before
termination. Other users of the selected GPU count toward the guard, but are
NEVER signalled. Only the command's fresh process group is terminated, including
ordinary descendants. Commands must not daemonize/escape that group via setsid
or setpgid; process groups are not a containment sandbox. Child stdout/stderr are
inherited verbatim. The required JSON output must not already exist.

Exit status preserves a finished command's code (128 + signal for signals).
Guard trips return 125, observation/evidence/cleanup failures 124, and launch
failures 127 (missing executable) or 126. Inspect JSON status to distinguish a
guard failure from a command that happens to use the same exit code.
"""

import argparse
import errno
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


GIB = 1024 ** 3
MIB = 1024 ** 2
SAMPLE_INTERVAL_SECONDS = 0.050
EXIT_GUARD = 125
EXIT_MONITOR = 124


def normalize_bdf(value):
    value = value.lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        value = "0000:" + value
    if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", value):
        raise argparse.ArgumentTypeError("expected PCI BDF dddd:bb:dd.f")
    return value


def counter_for_bdf(root, bdf):
    """Resolve once by PCI identity, never by an unstable DRM card number."""
    matches = []
    for card in sorted(root.iterdir()):
        if not re.fullmatch(r"card[0-9]+", card.name):
            continue
        try:
            device = (card / "device").resolve(strict=True)
        except FileNotFoundError:
            continue
        if device.name.lower() == bdf:
            matches.append((card / "device" / "mem_info_vram_used",
                            device / "mem_info_vram_used"))
    if len(matches) != 1:
        raise ValueError(f"expected one DRM card for PCI {bdf}, found {len(matches)}")
    return matches[0]


def read_sample(counter, started):
    raw = counter.read_text(encoding="ascii")
    if not re.fullmatch(r"[0-9]+\s*", raw, flags=re.ASCII):
        raise ValueError(f"invalid VRAM counter at {counter}: {raw!r}")
    return {"elapsed_seconds": time.monotonic() - started,
            "vram_used_bytes": int(raw), "raw": raw}


def observe_owned_child(process):
    """Observe without reaping: the unreaped PID pins our process-group ID.

    Never use Popen.poll() here: reaping the leader releases that identity and
    a subsequent killpg could target an unrelated, recycled process-group ID.
    ECHILD (including an external reaper) fails closed, even for signal 0.
    """
    if process.returncode is not None:
        raise ChildProcessError("leader already reaped; process-group ownership lost")
    try:
        return os.waitid(os.P_PID, process.pid,
                         os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError as exc:
        raise ChildProcessError("leader is not our unreaped child; "
                                "process-group ownership lost") from exc


def group_exists(process):
    observe_owned_child(process)
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_owned_group(process, grace_seconds):
    """Bounded TERM -> KILL, retaining the unreaped leader until all signals."""
    pgid = process.pid  # start_new_session=True makes PID == PGID.
    if pgid <= 1 or pgid == os.getpgrp():
        raise RuntimeError("refusing to signal a non-owned process group")
    sent = []

    def send(sig):
        observe_owned_child(process)
        try:
            os.killpg(pgid, sig)
            sent.append(signal.Signals(sig).name)
        except ProcessLookupError:
            pass

    if group_exists(process):
        send(signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if not group_exists(process):
                break
            time.sleep(min(SAMPLE_INTERVAL_SECONDS,
                           max(0, deadline - time.monotonic())))
        if group_exists(process):
            send(signal.SIGKILL)
    # This is the ONLY reap, after the final possible group signal. A zombie
    # leader can keep an otherwise empty group present through the grace period;
    # harmless TERM/KILL calls to that pinned group do not change its exit code.
    process.wait(timeout=1.0)
    # Orphaned zombies may retain the PGID until init reaps them. They cannot
    # hold VRAM, and are not a reason to wait indefinitely after SIGKILL.
    return sent


def shell_exit_code(returncode):
    return 128 - returncode if returncode < 0 else returncode


def run_guard(args, report):
    started = time.monotonic()
    process = None
    interrupted = []
    handlers = {}
    report.update({"schema_version": 1, "guard_kind": "sampled_not_hard_limit",
                   "limitations": "Device-wide sampling can miss short spikes or "
                   "overshoot; not a hard allocation cap. Escaped process groups "
                   "are not contained. Unrelated GPU users are never signalled.",
                   "pci_bdf": args.pci_bdf, "command": args.command,
                   "limit_bytes": args.limit_bytes,
                   "safety_margin_bytes": args.safety_margin_bytes,
                   "guard_limit_bytes": args.limit_bytes - args.safety_margin_bytes,
                   "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
                   "baseline_bytes": None, "peak_bytes": None,
                   "peak_delta_bytes": None, "samples": [],
                   "child_pid": None, "child_exit_code": None,
                   "termination_signals": [], "status": "monitor_error"})

    def sample(counter):
        item = read_sample(counter, started)
        report["samples"].append(item)
        value = item["vram_used_bytes"]
        if report["baseline_bytes"] is None:
            report["baseline_bytes"] = value
        report["peak_bytes"] = max(value, report["peak_bytes"] or 0)
        report["peak_delta_bytes"] = report["peak_bytes"] - report["baseline_bytes"]
        return value > report["guard_limit_bytes"]

    def on_signal(signum, _frame):
        if not interrupted:
            interrupted.append(signum)

    exit_code = EXIT_MONITOR
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, on_signal)
        display_counter, counter = counter_for_bdf(args._sysfs_root, args.pci_bdf)
        report["sysfs_counter"] = str(display_counter)
        report["resolved_counter"] = str(counter)
        if sample(counter):
            report["status"] = "limit_exceeded_before_launch"
            exit_code = EXIT_GUARD
        elif interrupted:
            report["status"] = "interrupted"
            exit_code = 128 + interrupted[0]
        else:
            try:
                process = subprocess.Popen(args.command, start_new_session=True)
            except OSError as exc:
                report["status"] = "launch_error"
                report["error"] = str(exc)
                exit_code = 127 if exc.errno == errno.ENOENT else 126
            else:
                report["child_pid"] = process.pid
                while True:
                    if interrupted:
                        report["status"] = "interrupted"
                        exit_code = 128 + interrupted[0]
                        break
                    # Includes an immediate post-launch and final post-exit
                    # sample: even a fast child cannot yield a fake monitor PASS.
                    if sample(counter):
                        report["status"] = "limit_exceeded"
                        exit_code = EXIT_GUARD
                        break
                    child_exit = observe_owned_child(process)
                    if child_exit is not None:
                        child_code = (child_exit.si_status
                                      if child_exit.si_code == os.CLD_EXITED
                                      else -child_exit.si_status)
                        report["status"] = "completed"
                        exit_code = shell_exit_code(child_code)
                        break
                    time.sleep(SAMPLE_INTERVAL_SECONDS)
    except (OSError, ValueError) as exc:
        report["status"] = "monitor_error"
        report["error"] = str(exc)
        exit_code = EXIT_MONITOR
    finally:
        if process is not None:
            try:
                report["termination_signals"] = terminate_owned_group(
                    process, args.terminate_timeout_seconds)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                report["status"] = "cleanup_error"
                report["cleanup_error"] = str(exc)
                exit_code = EXIT_MONITOR
            report["child_exit_code"] = process.returncode
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        report["elapsed_seconds"] = time.monotonic() - started
        report["exit_code"] = exit_code
    return exit_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pci-bdf", required=True, type=normalize_bdf)
    parser.add_argument("--out", required=True, type=Path,
                        help="new JSON evidence file (existing files are refused)")
    parser.add_argument("--limit-bytes", type=int, default=24 * GIB,
                        help="total device-wide cap before margin (default: 24 GiB)")
    parser.add_argument("--safety-margin-bytes", type=int, default=512 * MIB,
                        help="terminate above cap minus margin (default: 512 MiB)")
    parser.add_argument("--terminate-timeout-seconds", type=float, default=2.0,
                        help="SIGTERM grace, then SIGKILL (default: 2 seconds)")
    parser.add_argument("--_sysfs-root", type=Path, default=Path("/sys/class/drm"),
                        help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command or args.command[0] != "--" or len(args.command) == 1:
        parser.error("supply a command after --")
    args.command = args.command[1:]
    if args.limit_bytes <= 0 or not 0 <= args.safety_margin_bytes < args.limit_bytes:
        parser.error("require 0 <= safety margin < positive limit")
    if not 0 <= args.terminate_timeout_seconds <= 60:
        parser.error("termination timeout must be finite and between 0 and 60 seconds")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        # O_EXCL refuses regular files, symlinks, and concurrent writers alike.
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        print(f"VRAM guard: cannot create evidence file: {exc}", file=sys.stderr)
        return EXIT_MONITOR
    print("VRAM guard: sampled device-wide observation, NOT a hard physical "
          "limit; only the new command process group can be terminated.", file=sys.stderr)
    report = {}
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            result = run_guard(args, report)
            json.dump(report, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        print(f"VRAM guard: evidence write failed: {exc}", file=sys.stderr)
        return EXIT_MONITOR
    print(f"VRAM guard: {report['status']}; peak={report['peak_bytes']} "
          f"guard={report['guard_limit_bytes']} bytes; evidence={args.out}", file=sys.stderr)
    return result


if __name__ == "__main__":
    sys.exit(main())
