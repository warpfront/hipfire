#!/usr/bin/env bash
# Sourced helpers shared by gap-cell.sh / gap-job.sh / gap-wait.sh. Detection only: nothing here kills anything.
export PYTHONDONTWRITEBYTECODE=1

# gap_owned_pids PREFIX: pids (other than this shell) whose environment carries HIPFIRE_HOME=PREFIX*. Every model
# process of a cell is started with HIPFIRE_HOME=<cell dir>/home/.hipfire, so this finds exactly the processes the
# harness owns, including any that left their process group. Unreadable /proc entries (other users) are skipped.
gap_owned_pids() {
  local prefix=$1 d p
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    [ "$p" = "$$" ] && continue
    [ -r "$d/environ" ] || continue
    if grep -a -z -q -F -e "HIPFIRE_HOME=$prefix" "$d/environ" 2>/dev/null; then printf '%s\n' "$p"; fi
  done
}

# gap_describe PID...: one ps line per pid still present.
gap_describe() {
  [ "$#" -gt 0 ] || return 0
  ps -o pid=,ppid=,pgid=,sid=,stat=,etimes=,args= -p "$(IFS=,; echo "$*")" 2>/dev/null || true
}

# gap_live PID...: the subset of pids that exist and are not zombies.
gap_live() {
  local p s
  for p in "$@"; do
    s=$(ps -o stat= -p "$p" 2>/dev/null) || continue
    case "$s" in ''|Z*) ;; *) printf '%s\n' "$p" ;; esac
  done
}

# gap_survivors CPID PREFIX: live pids that belong to this cell: its process group, its session, or anything whose
# environment carries HIPFIRE_HOME=PREFIX*. CPID may be empty (model never started).
gap_survivors() {
  local cpid=$1 prefix=$2
  {
    if [ -n "$cpid" ]; then pgrep -g "$cpid" 2>/dev/null; pgrep -s "$cpid" 2>/dev/null; fi
    gap_owned_pids "$prefix"
  } | sort -un | while read -r p; do [ "$p" = "$$" ] || printf '%s\n' "$p"; done | { mapfile -t all; [ "${#all[@]}" -gt 0 ] && gap_live "${all[@]}"; true; }
}

# gap_vendor_ok BDF: 0 iff the PCI device still reads AMD Navi31 (0x1002:0x744c).
gap_vendor_ok() {
  local v d
  v=$(cat "/sys/bus/pci/devices/$1/vendor" 2>/dev/null || echo missing)
  d=$(cat "/sys/bus/pci/devices/$1/device" 2>/dev/null || echo missing)
  [ "$v" = 0x1002 ] && [ "$d" = 0x744c ]
}
