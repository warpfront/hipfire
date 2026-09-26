#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# gpu-lock.sh — GPU mutex for multi-agent coordination
# Source this in an agent session:  source scripts/gpu-lock.sh
# Then:  gpu_acquire "model-ingestion" && { run tests; gpu_release; }
# Per-card: gpu_acquire "model-ingestion" GPU-05f92432f2312a0e
#           gpu_acquire "model-ingestion" 0000:99:00.0   (card without a UUID)
# A single UUID in HIP_VISIBLE_DEVICES is inferred when no second argument is
# given; otherwise no card uses the legacy global /tmp/hipfire-gpu.lock.
# Per-card locks use the daemon's keys: gpu-GPU-<16 hex>.lock for a card with
# a UUID, gpu-pci-<dddd:bb:dd.f>.lock for one without (its UUID reads 0/XX).
# They live in HIPFIRE_LOCK_DIR if set, else writable /run/lock/hipfire, else
# /tmp/hipfire-locks (the daemon's same policy).
#
# Backed by flock(1): the lock is held on an open file descriptor, so the
# Linux kernel releases it automatically when the holding process dies for
# ANY reason — kill -9, crash, OOM, agent cancellation, terminal close.
# Stale locks are therefore structurally impossible: there is nothing to
# clean up by hand.
#
# IMPORTANT: never `rm` the lockfile. Unlinking a file that a holder has
# flock'd lets the next acquirer create a *different* inode and lock that
# too — yielding two simultaneous holders. The file is a permanent fixture;
# recovery happens in the kernel, not by deleting the file.

LOCKFILE="${HIPFIRE_GPU_LOCKFILE:-/tmp/hipfire-gpu.lock}"
GPU_LOCK_DIR=""
GPU_ACTIVE_LOCKFILE=""
POLL_INTERVAL="${GPU_POLL_INTERVAL:-5}"          # cadence of "busy" messages (s)
(( POLL_INTERVAL < 1 )) && POLL_INTERVAL=1        # 0 would make flock -w non-blocking → spin
GPU_LOCK_TIMEOUT="${GPU_LOCK_TIMEOUT:-1800}"     # hard cap (s); 0 = wait forever
GPU_LOCK_FD=""                                   # set by gpu_acquire (auto-alloc)

# Print the daemon's lock identity for a GPU-<hex> UUID or a PCI address.
gpu_lock_identity() {
    local card="$1"
    if [[ "$card" =~ ^[Gg][Pp][Uu]-([[:xdigit:]]{1,16})$ ]]; then
        local hex="${BASH_REMATCH[1],,}"
        while [[ "$hex" == 0* ]]; do hex="${hex#0}"; done
        if [ -z "$hex" ]; then
            echo "[gpu-lock] $card is the no-UUID placeholder; pass the card's PCI address" >&2
            return 3
        fi
        while [ "${#hex}" -lt 16 ]; do hex="0$hex"; done
        printf 'GPU-%s\n' "$hex"
    elif [[ "$card" =~ ^(([[:xdigit:]]{1,8}):)?([[:xdigit:]]{1,2}):([[:xdigit:]]{1,2})\.([0-7])$ ]]; then
        printf 'pci-%04x:%02x:%02x.%x\n' "0x${BASH_REMATCH[2]:-0}" "0x${BASH_REMATCH[3]}" \
            "0x${BASH_REMATCH[4]}" "0x${BASH_REMATCH[5]}"
    else
        echo "[gpu-lock] expected GPU-<uuid> or a PCI address [DDDD:]BB:DD.F: $card" >&2
        return 3
    fi
}

gpu_acquire() {
    local agent_name="${1:?usage: gpu_acquire <agent-name> [GPU-uuid|PCI-address]}"
    local card="${2:-}"
    local identity=""
    local requested_lockfile
    if [ -z "$card" ] && [[ "${HIP_VISIBLE_DEVICES:-}" =~ ^GPU-[[:xdigit:]]{16}$ ]]; then
        card="$HIP_VISIBLE_DEVICES"
    fi
    if [ -n "$card" ]; then
        identity=$(gpu_lock_identity "$card") || return 3
        if [ "${HIPFIRE_LOCK_DIR+x}" = x ]; then
            GPU_LOCK_DIR="$HIPFIRE_LOCK_DIR"
            if [[ "$GPU_LOCK_DIR" != /* ]]; then
                echo "[gpu-lock] HIPFIRE_LOCK_DIR must be absolute: $GPU_LOCK_DIR" >&2
                return 3
            fi
            if [ ! -d "$GPU_LOCK_DIR" ]; then
                mkdir -p "$GPU_LOCK_DIR" || return 3
                chmod 1777 "$GPU_LOCK_DIR" || return 3
            fi
            if [ ! -w "$GPU_LOCK_DIR" ] || [ ! -x "$GPU_LOCK_DIR" ]; then
                echo "[gpu-lock] lock directory is not writable: $GPU_LOCK_DIR" >&2
                return 3
            fi
        else
            for GPU_LOCK_DIR in /run/lock/hipfire /tmp/hipfire-locks; do
                if [ ! -d "$GPU_LOCK_DIR" ]; then
                    mkdir -p "$GPU_LOCK_DIR" 2>/dev/null || continue
                    chmod 1777 "$GPU_LOCK_DIR" 2>/dev/null || continue
                fi
                [ -w "$GPU_LOCK_DIR" ] && [ -x "$GPU_LOCK_DIR" ] && break
            done
            if [ ! -w "$GPU_LOCK_DIR" ] || [ ! -x "$GPU_LOCK_DIR" ]; then
                echo "[gpu-lock] no writable machine-wide lock directory" >&2
                return 3
            fi
        fi
        echo "[gpu-lock] directory=$GPU_LOCK_DIR"
        requested_lockfile="$GPU_LOCK_DIR/gpu-${identity}.lock"
    else
        requested_lockfile="$LOCKFILE"
    fi

    if ! command -v flock >/dev/null 2>&1; then
        echo "[gpu-lock] FATAL: flock(1) not found — refusing to run unlocked" >&2
        return 3
    fi

    # Reentrancy: an ancestor in this process tree already holds the lock.
    # Recognise it and no-op so nested gates (e.g. pp-gate) don't deadlock on
    # their own parent's reservation.
    if [ -n "${HIPFIRE_GPU_LOCK_OWNER:-}" ]; then
        if [ "${HIPFIRE_GPU_LOCK_PATH:-}" != "$requested_lockfile" ]; then
            echo "[gpu-lock] FATAL: already holding ${HIPFIRE_GPU_LOCK_PATH:-another lock}; cannot silently reserve $requested_lockfile" >&2
            return 3
        fi
        echo "[gpu-lock] reentrant: already held by ancestor pid=${HIPFIRE_GPU_LOCK_OWNER}"
        return 0
    fi
    GPU_ACTIVE_LOCKFILE="$requested_lockfile"

    # Open (create if missing) WITHOUT truncating, so we don't clobber a live
    # holder's metadata while we wait. Bash auto-allocates a free fd into
    # GPU_LOCK_FD; it persists in this shell until gpu_release closes it.
    exec {GPU_LOCK_FD}<>"$GPU_ACTIVE_LOCKFILE"
    chmod 666 "$GPU_ACTIVE_LOCKFILE" 2>/dev/null || :

    local waited=0
    until flock -w "$POLL_INTERVAL" "$GPU_LOCK_FD"; do
        waited=$(( waited + POLL_INTERVAL ))
        local holder
        holder=$(cat "$GPU_ACTIVE_LOCKFILE" 2>/dev/null || echo unknown)
        if [ "$GPU_LOCK_TIMEOUT" -gt 0 ] && [ "$waited" -ge "$GPU_LOCK_TIMEOUT" ]; then
            echo "[gpu-lock] TIMEOUT after ${waited}s; holder still alive: ${holder}" >&2
            exec {GPU_LOCK_FD}>&-
            GPU_LOCK_FD=""
            return 2
        fi
        echo "[gpu-lock] busy: ${holder} — waited ${waited}s, still waiting…"
    done

    # We hold it. Record metadata (truncate + write via a separate redirect;
    # this does not disturb the flock held on $GPU_LOCK_FD).
    printf '%s %s host=%s acquired=%s\n' \
        "$$" "$agent_name" "$(hostname)" "$(date -Is 2>/dev/null || date)" > "$GPU_ACTIVE_LOCKFILE"
    export HIPFIRE_GPU_LOCK_OWNER="$$"
    export HIPFIRE_GPU_LOCK_PATH="$GPU_ACTIVE_LOCKFILE"
    echo "[gpu-lock] acquired by ${agent_name}"
    return 0
}

gpu_release() {
    # Only the process that actually acquired may release. A child that merely
    # inherited HIPFIRE_GPU_LOCK_OWNER from an ancestor must not release it.
    if [ -z "${HIPFIRE_GPU_LOCK_OWNER:-}" ]; then
        echo "[gpu-lock] no lock held"
        return 0
    fi
    if [ "$HIPFIRE_GPU_LOCK_OWNER" != "$$" ]; then
        return 0   # reentrant child — the ancestor owns it, leave it be
    fi

    if [ -n "${GPU_LOCK_FD:-}" ]; then
        flock -u "$GPU_LOCK_FD" 2>/dev/null
        exec {GPU_LOCK_FD}>&- 2>/dev/null
        GPU_LOCK_FD=""
    fi
    unset HIPFIRE_GPU_LOCK_OWNER HIPFIRE_GPU_LOCK_PATH
    GPU_ACTIVE_LOCKFILE=""
    # NB: never rm "$LOCKFILE" — see header.
    echo "[gpu-lock] released"
}

gpu_status() {
    local path="${GPU_ACTIVE_LOCKFILE:-$LOCKFILE}"
    if [ ! -e "$path" ]; then
        echo "gpu is free"
        return 0
    fi
    # Probe with a non-blocking flock on a scratch fd. If we can take it,
    # nobody is holding it — any leftover file content is cosmetic/stale.
    local probe_fd
    exec {probe_fd}<>"$path"
    if flock -n "$probe_fd"; then
        flock -u "$probe_fd"
        exec {probe_fd}>&-
        echo "gpu is free"
    else
        exec {probe_fd}>&-
        echo "gpu BUSY: $(cat "$path" 2>/dev/null)"
    fi
}
