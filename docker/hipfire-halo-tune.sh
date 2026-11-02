#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# hipfire-halo-tune: opt-in host tuner for Strix Halo (gfx1151).
# POSIX sh + coreutils only. See --help.

PROG=hipfire-halo-tune
SYSFS=/sys
STATE_DIR=${HIPFIRE_HALO_TUNE_STATE:-/var/lib/hipfire-halo-tune}
STATE_FILE=$STATE_DIR/state
PERF_LEVEL=${HIPFIRE_HALO_PERF_LEVEL:-auto}
NO_HALO_MSG='no Strix Halo (gfx1151) GPU found; nothing changed'

NL='
'
TAB=$(printf '\t')

usage() {
    cat <<'EOF'
Usage: hipfire-halo-tune.sh [status]
       hipfire-halo-tune.sh apply [--dry-run]
       hipfire-halo-tune.sh restore
       hipfire-halo-tune.sh -h | --help

Opt-in host tuner for a Strix Halo (gfx1151) machine. It puts the machine in
the power state hipfire's published Halo speeds are measured in: ACPI platform
profile "performance", GPU performance level "auto", GPU power profile not
"POWER_SAVING", and the GPU power cap at its default.

Commands:
  status           Show the current values and what apply would change.
                   Read-only; this is the default command.
  apply            Change only the values that differ. Needs root.
  apply --dry-run  Print the planned changes; writes nothing, not even the
                   saved-state file.
  restore          Put the values saved by apply back, then delete the saved
                   state. Needs root. With no saved state it does nothing and
                   exits 0.

WARNING: apply is a system-wide mode, not a per-process setting. It raises
package power limits, fan noise and power draw for everything on the machine.
What it touches, and nothing else:
  * ACPI platform profile -> performance (only if the firmware lists it):
      /sys/firmware/acpi/platform_profile, else
      /sys/class/platform-profile/*/profile
  * GPU DPM performance level (power_dpm_force_performance_level) ->
      ${HIPFIRE_HALO_PERF_LEVEL:-auto}; a "manual" level (set by another tool,
      for example an FCLK pin) is reported and left unchanged
  * GPU power profile (pp_power_profile_mode): when the active profile is
      POWER_SAVING -> BOOTUP_DEFAULT
  * GPU power cap (hwmon power1_cap): when below power1_cap_default ->
      the default

Each change is logged as '<path>: <before> -> <after>'. Absent or unwritable
attributes are skipped with a note. The original values are saved before the
first write to ${HIPFIRE_HALO_TUNE_STATE:-/var/lib/hipfire-halo-tune}/state; a
second apply never rewrites an existing state file.

Environment:
  HIPFIRE_HALO_PERF_LEVEL   auto (default) or high. auto is the published
                            default. high is optional and NOT the published
                            default: it pins the GPU at its top DPM levels.
  HIPFIRE_HALO_TUNE_STATE   Directory for the saved state
                            (default /var/lib/hipfire-halo-tune).

Limits:
  * Changes do not survive a reboot; run apply again after each boot, or
    restore to undo them sooner.
  * It never touches BIOS settings, SMU settings or FCLK.
  * Rootless podman cannot write host sysfs. Use rootful docker or podman, or
    run it on the host directly:
      sudo docker/hipfire-halo-tune.sh apply
    Inside the image: hipfire-halo-tune (privileged, root), for example the
    compose "halo-tune" profile.

Exit codes: 0 done (also: nothing to do, status without a Halo), 1 error or
a failed write, 2 usage error or no Strix Halo GPU found by apply.
EOF
}

die() {
    _code=$1
    shift
    printf '%s: %s\n' "$PROG" "$*" >&2
    exit "$_code"
}

usage_error() {
    printf '%s: %s\n' "$PROG" "$*" >&2
    printf 'Try: %s --help\n' "$PROG" >&2
    exit 2
}

# ---------------------------------------------------------------- helpers

# trim_var TEXT -> T without leading/trailing whitespace.
trim_var() {
    T=$1
    while :; do
        case $T in
        [[:space:]]*) T=${T#?} ;;
        *[[:space:]]) T=${T%?} ;;
        *) break ;;
        esac
    done
}

# read_attr FILE -> RA (trimmed, non-empty); fails when unreadable or empty.
read_attr() {
    RA=''
    [ -r "$1" ] || return 1
    _ra=$(cat "$1" 2>/dev/null) || return 1
    trim_var "$_ra"
    [ -n "$T" ] || return 1
    RA=$T
}

# write_attr FILE VALUE
write_attr() {
    (printf '%s\n' "$2" >"$1") 2>/dev/null
}

is_uint() {
    case $1 in
    '' | *[!0-9]*) return 1 ;;
    esac
}

note() {
    NOTES="$NOTES  note: $*$NL"
}

info() {
    INFO="$INFO  $*$NL"
}

# ------------------------------------------------------ Halo detection
#
# Mirrors the doctor: a KFD topology node with simd_count > 0 and
# gfx_target_version 110501 (gfx1151) gives the PCI address
# (location_id = bus << 8 | device << 3 | function); several Halo nodes are
# ordered by PCI address and the first one wins. When the KFD topology is not
# visible, fall back to the amdgpu-bound Strix Halo PCI device in sysfs.

HALO_DEV=''
HALO_BDF=''
HALO_SRC=''

find_halo() {
    HALO_DEV=''
    HALO_BDF=''
    HALO_SRC=''
    _pci="$SYSFS/bus/pci/devices"
    _list=''
    if [ -d "$SYSFS/class/kfd/kfd/topology/nodes" ]; then
        for _node in "$SYSFS/class/kfd/kfd/topology/nodes"/*; do
            case ${_node##*/} in
            '' | *[!0-9]*) continue ;;
            esac
            [ -r "$_node/properties" ] || continue
            _simd=0
            _loc=''
            _dom=0
            _ver=''
            while read -r _k _v; do
                case $_k in
                simd_count) _simd=$_v ;;
                location_id) _loc=$_v ;;
                domain) _dom=$_v ;;
                gfx_target_version) _ver=$_v ;;
                esac
            done <"$_node/properties" 2>/dev/null
            is_uint "$_simd" || continue
            [ "$_simd" -gt 0 ] || continue
            [ "$_ver" = 110501 ] || continue
            is_uint "$_loc" || continue
            is_uint "$_dom" || continue
            _bdf=$(printf '%04x:%02x:%02x.%x' "$_dom" \
                "$(( (_loc >> 8) & 255 ))" "$(( (_loc >> 3) & 31 ))" "$(( _loc & 7 ))")
            _list="$_list$_bdf$NL"
        done
        if [ -n "$_list" ]; then
            for _bdf in $(printf '%s' "$_list" | LC_ALL=C sort); do
                if [ -d "$_pci/$_bdf" ]; then
                    HALO_BDF=$_bdf
                    HALO_DEV="$_pci/$_bdf"
                    HALO_SRC='KFD topology'
                    return 0
                fi
            done
        fi
    fi
    # Fallback: the amdgpu-bound PCI device whose IP discovery table lists a
    # GC (graphics) block at version 11.5.1, the gfx1151 graphics IP. Layout
    # (amdgpu_discovery.c): <pci dev>/ip_discovery/die/<die id>/GC/<instance>/
    # with decimal major, minor and revision files ("GC" links to the hw id).
    for _d in "$_pci"/*; do
        [ -d "$_d" ] || continue
        read_attr "$_d/vendor" || continue
        [ "$RA" = 0x1002 ] || continue
        for _gc in "$_d"/ip_discovery/die/*/GC/*; do
            [ -d "$_gc" ] || continue
            read_attr "$_gc/major" && [ "$RA" = 11 ] || continue
            read_attr "$_gc/minor" && [ "$RA" = 5 ] || continue
            read_attr "$_gc/revision" && [ "$RA" = 1 ] || continue
            HALO_BDF=${_d##*/}
            HALO_DEV=$_d
            HALO_SRC='amdgpu IP discovery (GC 11.5.1)'
            return 0
        done
    done
    return 1
}

# ----------------------------------------------------- reading values

# parse_profile_text TEXT -> PM_ACTIVE_IDX PM_ACTIVE_NAME PM_BOOT_IDX.
# Rows look like '  0 BOOTUP_DEFAULT*: ...'; the '*' on the name marks the
# active one (same reading as the doctor).
parse_profile_text() {
    PM_ACTIVE_IDX=''
    PM_ACTIVE_NAME=''
    PM_BOOT_IDX=''
    set -f
    while IFS= read -r _line; do
        trim_var "$_line"
        _t=$T
        case $_t in
        [0-9]*) ;;
        *) continue ;;
        esac
        _idx=${_t%%[!0-9]*}
        _rest=${_t#"$_idx"}
        case $_rest in
        [[:space:]]*) ;;
        *) continue ;;
        esac
        _field=${_rest%%:*}
        _star=0
        case $_field in
        *'*'*) _star=1 ;;
        esac
        _field=$(printf '%s' "$_field" | tr -d '*')
        _name=''
        for _w in $_field; do
            _name=$_w
            break
        done
        [ -n "$_name" ] || continue
        if [ "$_name" = BOOTUP_DEFAULT ] && [ -z "$PM_BOOT_IDX" ]; then
            PM_BOOT_IDX=$_idx
        fi
        if [ "$_star" = 1 ] && [ -z "$PM_ACTIVE_IDX" ]; then
            PM_ACTIVE_IDX=$_idx
            PM_ACTIVE_NAME=$_name
        fi
    done <<EOF
$1
EOF
    set +f
}

# cur_value KIND PATH -> CUR_W (comparable/writable value) CUR_D (display).
cur_value() {
    CUR_W=''
    CUR_D=''
    case $1 in
    power_profile)
        [ -r "$2" ] || return 1
        _txt=$(cat "$2" 2>/dev/null) || return 1
        parse_profile_text "$_txt"
        [ -n "$PM_ACTIVE_IDX" ] || return 1
        CUR_W=$PM_ACTIVE_IDX
        CUR_D=$PM_ACTIVE_NAME
        ;;
    *)
        read_attr "$2" || return 1
        CUR_W=$RA
        CUR_D=$RA
        ;;
    esac
}

# ------------------------------------------------------------ planning
#
# Planning only reads. PLAN lines, TAB-separated:
#   kind, original value, original display, new value, new display, path

PLAN=''
SNAPSHOT=''
NOTES=''
INFO=''
CHECK_W=0

plan_add() {
    PLAN="$PLAN$1$TAB$2$TAB$3$TAB$4$TAB$5$TAB$6$NL"
}

# snap_add KIND ORIGINAL DISPLAY PATH: every readable original, whether or not
# it differs from the target, so the first saved state can restore any later
# apply. Same field order as the state file.
snap_add() {
    SNAPSHOT="$SNAPSHOT$1$TAB$2$TAB$3$TAB$4$NL"
}

plan_platform() {
    _file=''
    _choices_file=''
    if read_attr "$SYSFS/firmware/acpi/platform_profile"; then
        _file="$SYSFS/firmware/acpi/platform_profile"
        _choices_file="$SYSFS/firmware/acpi/platform_profile_choices"
        _cur=$RA
    else
        for _d in "$SYSFS"/class/platform-profile/*; do
            if read_attr "$_d/profile"; then
                _file="$_d/profile"
                _choices_file="$_d/choices"
                _cur=$RA
                break
            fi
        done
    fi
    if [ -z "$_file" ]; then
        note "platform profile: neither $SYSFS/firmware/acpi/platform_profile nor $SYSFS/class/platform-profile/*/profile is readable; skipped"
        return
    fi
    _choices=''
    if read_attr "$_choices_file"; then
        _choices=$RA
    fi
    info "$_file: $_cur${_choices:+ (choices: $_choices)}"
    snap_add platform_profile "$_cur" "$_cur" "$_file"
    [ "$_cur" = performance ] && return
    if [ -z "$_choices" ]; then
        note "$_file: choices unreadable ($_choices_file); skipped"
        return
    fi
    case " $_choices " in
    *' performance '*) ;;
    *)
        note "$_file: 'performance' is not among the choices ($_choices); skipped"
        return
        ;;
    esac
    if [ "$CHECK_W" = 1 ] && [ ! -w "$_file" ]; then
        note "$_file: not writable; skipped"
        return
    fi
    plan_add platform_profile "$_cur" "$_cur" performance performance "$_file"
}

plan_perf() {
    _file="$HALO_DEV/power_dpm_force_performance_level"
    if ! read_attr "$_file"; then
        note "$_file: absent or unreadable; skipped"
        return
    fi
    info "$_file: $RA (target: $PERF_LEVEL)"
    snap_add perf_level "$RA" "$RA" "$_file"
    [ "$RA" = "$PERF_LEVEL" ] && return
    # manual carries clock masks (an FCLK pin, say) this script cannot read back
    # or restore, so it is someone else's setting: report it, never override it.
    if [ "$RA" = manual ]; then
        note "$_file: manual (set by another tool, for example an FCLK pin); left unchanged"
        return
    fi
    if [ "$CHECK_W" = 1 ] && [ ! -w "$_file" ]; then
        note "$_file: not writable; skipped"
        return
    fi
    plan_add perf_level "$RA" "$RA" "$PERF_LEVEL" "$PERF_LEVEL" "$_file"
}

plan_profile() {
    _file="$HALO_DEV/pp_power_profile_mode"
    if ! cur_value power_profile "$_file"; then
        note "$_file: absent, unreadable or no active profile; skipped"
        return
    fi
    info "$_file: $CUR_D (index $CUR_W)"
    snap_add power_profile "$CUR_W" "$CUR_D" "$_file"
    [ "$CUR_D" = POWER_SAVING ] || return
    if [ -z "$PM_BOOT_IDX" ]; then
        note "$_file: no BOOTUP_DEFAULT entry to switch to; skipped"
        return
    fi
    if [ "$CHECK_W" = 1 ] && [ ! -w "$_file" ]; then
        note "$_file: not writable; skipped"
        return
    fi
    plan_add power_profile "$CUR_W" "$CUR_D" "$PM_BOOT_IDX" BOOTUP_DEFAULT "$_file"
}

plan_cap() {
    _hw=''
    for _d in "$HALO_DEV"/hwmon/*; do
        [ -d "$_d" ] || continue
        [ -n "$_hw" ] || _hw=$_d
        if read_attr "$_d/name" && [ "$RA" = amdgpu ]; then
            _hw=$_d
            break
        fi
    done
    if [ -z "$_hw" ]; then
        note "$HALO_DEV/hwmon: no hwmon directory; power cap skipped"
        return
    fi
    _capf="$_hw/power1_cap"
    _deff="$_hw/power1_cap_default"
    if ! read_attr "$_capf" || ! is_uint "$RA"; then
        note "$_capf: absent or unreadable; skipped"
        return
    fi
    _cap=$RA
    if ! read_attr "$_deff" || ! is_uint "$RA"; then
        note "$_deff: absent or unreadable; power cap skipped"
        return
    fi
    _def=$RA
    info "$_capf: $_cap (default: $_def)"
    snap_add power_cap "$_cap" "$_cap" "$_capf"
    [ "$_cap" -lt "$_def" ] || return
    if [ "$CHECK_W" = 1 ] && [ ! -w "$_capf" ]; then
        note "$_capf: not writable; skipped"
        return
    fi
    plan_add power_cap "$_cap" "$_cap" "$_def" "$_def" "$_capf"
}

plan_all() {
    PLAN=''
    NOTES=''
    INFO=''
    SNAPSHOT=''
    plan_platform
    plan_perf
    plan_profile
    plan_cap
}

print_plan() {
    [ -n "$PLAN" ] || return 0
    while IFS=$TAB read -r _kind _ow _od _nw _nd _path; do
        [ -n "$_kind" ] || continue
        printf '%s: %s -> %s\n' "$_path" "$_od" "$_nd"
    done <<EOF
$PLAN
EOF
}

# ------------------------------------------------------- saved state
#
# State lines, TAB-separated: kind, original value, original display, path
# (the SNAPSHOT taken by the planners).
# The first snapshot is immutable: written (atomically) before the first
# change, and never rewritten or appended to while it exists.

save_state() {
    if [ -f "$STATE_FILE" ]; then
        printf 'keeping the saved originals in %s (not overwritten)\n' "$STATE_FILE"
        return 0
    fi
    # Only called when PLAN is non-empty. SNAPSHOT holds every readable
    # original, so a later apply that changes more (auto -> high, say) is
    # still restorable from this first snapshot.
    _new=$SNAPSHOT
    mkdir -p "$STATE_DIR" || {
        printf '%s: cannot create state directory %s\n' "$PROG" "$STATE_DIR" >&2
        return 1
    }
    _tmp="$STATE_FILE.tmp.$$"
    if ! { printf '%s\n' '# hipfire-halo-tune saved values v1'; printf '%s' "$_new"; } >"$_tmp" ||
        ! mv -f "$_tmp" "$STATE_FILE"; then
        printf '%s: cannot write %s\n' "$PROG" "$STATE_FILE" >&2
        rm -f "$_tmp"
        return 1
    fi
}

# ----------------------------------------------------------- writing

# set_value KIND VALUE PATH -> 0 written and verified, 3 not writable,
# 1 write failed or readback differs.
set_value() {
    [ -w "$3" ] || return 3
    if ! write_attr "$3" "$2"; then
        SET_ERR='write failed'
        return 1
    fi
    if ! cur_value "$1" "$3"; then
        SET_ERR='could not read the value back'
        return 1
    fi
    if [ "$CUR_W" != "$2" ]; then
        SET_ERR="readback shows '$CUR_D', expected '$2'"
        return 1
    fi
    return 0
}

need_root() {
    if [ "$(id -u 2>/dev/null)" != 0 ]; then
        die 1 "$1 must run as root (try: sudo $0 $1)"
    fi
}

# --------------------------------------------------------- commands

cmd_status() {
    if ! find_halo; then
        printf '%s\n' "$NO_HALO_MSG"
        return 0
    fi
    CHECK_W=0
    plan_all
    printf 'Strix Halo (gfx1151) GPU: %s (found via %s)\n' "$HALO_BDF" "$HALO_SRC"
    printf 'current values:\n'
    printf '%s' "$INFO"
    if [ -n "$PLAN" ]; then
        printf 'apply would change:\n'
        print_plan | while IFS= read -r _l; do printf '  %s\n' "$_l"; done
    else
        printf 'apply would change nothing\n'
    fi
    if [ -n "$NOTES" ]; then
        printf '%s' "$NOTES"
    fi
    if [ -f "$STATE_FILE" ]; then
        printf 'saved state: %s (run restore to put the original values back)\n' "$STATE_FILE"
    else
        printf 'saved state: none (%s)\n' "$STATE_FILE"
    fi
    return 0
}

cmd_apply() {
    if ! find_halo; then
        printf '%s\n' "$NO_HALO_MSG" >&2
        exit 2
    fi
    if [ "$DRY" = 0 ]; then
        need_root apply
        CHECK_W=1
    else
        if [ "$(id -u 2>/dev/null)" = 0 ]; then
            CHECK_W=1
        else
            CHECK_W=0
        fi
    fi
    plan_all
    if [ -z "$PLAN" ]; then
        printf 'Strix Halo GPU %s: nothing to change\n' "$HALO_BDF"
        printf '%s' "$NOTES"
        return 0
    fi
    if [ "$DRY" = 1 ]; then
        printf 'dry-run: planned changes (nothing written, including the saved state):\n'
        print_plan
        printf '%s' "$NOTES"
        if [ "$CHECK_W" = 0 ]; then
            printf '  note: not running as root; writability was not checked and apply needs root\n'
        fi
        return 0
    fi
    save_state || die 1 "could not save the original values; nothing changed"
    _failed=0
    _done=0
    while IFS=$TAB read -r _kind _ow _od _nw _nd _path; do
        [ -n "$_kind" ] || continue
        set_value "$_kind" "$_nw" "$_path"
        case $? in
        0)
            printf '%s: %s -> %s\n' "$_path" "$_od" "$_nd"
            _done=$((_done + 1))
            ;;
        3)
            NOTES="$NOTES  note: $_path: not writable; skipped$NL"
            ;;
        *)
            printf '%s: %s: %s\n' "$PROG" "$_path" "$SET_ERR" >&2
            _failed=$((_failed + 1))
            ;;
        esac
    done <<EOF
$PLAN
EOF
    printf '%s' "$NOTES"
    printf 'applied %s change(s); originals saved in %s\n' "$_done" "$STATE_FILE"
    printf 'changes last until reboot; undo sooner with: %s restore\n' "$PROG"
    [ "$_failed" = 0 ] || return 1
    return 0
}

cmd_restore() {
    if [ ! -f "$STATE_FILE" ]; then
        printf 'no saved state at %s; nothing to restore\n' "$STATE_FILE"
        return 0
    fi
    need_root restore
    _rev=''
    while IFS= read -r _line; do
        case $_line in
        '' | '#'*) continue ;;
        esac
        _rev="$_line$NL$_rev"
    done <"$STATE_FILE"
    _failed=0
    _done=0
    while IFS=$TAB read -r _kind _ow _od _path; do
        [ -n "$_kind" ] || continue
        if ! cur_value "$_kind" "$_path"; then
            printf '  note: %s: absent or unreadable; skipped\n' "$_path"
            continue
        fi
        [ "$CUR_W" = "$_ow" ] && continue
        _before=$CUR_D
        set_value "$_kind" "$_ow" "$_path"
        case $? in
        0)
            printf '%s: %s -> %s\n' "$_path" "$_before" "$_od"
            _done=$((_done + 1))
            ;;
        3)
            printf '%s: %s: not writable\n' "$PROG" "$_path" >&2
            _failed=$((_failed + 1))
            ;;
        *)
            printf '%s: %s: %s\n' "$PROG" "$_path" "$SET_ERR" >&2
            _failed=$((_failed + 1))
            ;;
        esac
    done <<EOF
$_rev
EOF
    if [ "$_failed" != 0 ]; then
        printf '%s: %s value(s) could not be restored; keeping %s\n' \
            "$PROG" "$_failed" "$STATE_FILE" >&2
        return 1
    fi
    if ! rm -f "$STATE_FILE"; then
        printf '%s: restored, but could not delete %s\n' "$PROG" "$STATE_FILE" >&2
        return 1
    fi
    rmdir "$STATE_DIR" 2>/dev/null
    printf 'restored %s value(s); saved state removed\n' "$_done"
    return 0
}

# -------------------------------------------------------------- main

CMD=${1:-status}
if [ $# -gt 0 ]; then
    shift
fi
DRY=0
case $CMD in
-h | --help | help)
    usage
    exit 0
    ;;
status | restore)
    for _a in "$@"; do
        case $_a in
        -h | --help)
            usage
            exit 0
            ;;
        *) usage_error "unexpected argument to $CMD: $_a" ;;
        esac
    done
    ;;
apply)
    for _a in "$@"; do
        case $_a in
        --dry-run) DRY=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) usage_error "unexpected argument to apply: $_a" ;;
        esac
    done
    ;;
*) usage_error "unknown command: $CMD" ;;
esac

case $CMD in
status | apply)
    case $PERF_LEVEL in
    auto | high) ;;
    *) usage_error "HIPFIRE_HALO_PERF_LEVEL must be auto or high (got '$PERF_LEVEL')" ;;
    esac
    ;;
esac

case $CMD in
status) cmd_status ;;
apply) cmd_apply ;;
restore) cmd_restore ;;
esac
exit $?
