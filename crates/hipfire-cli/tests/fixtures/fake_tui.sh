#!/bin/sh
# Test-only stand-in for `hipfire-tui`: records its arguments, one per line, in
# the file named by FAKE_TUI_ARGS. Checked in and executed in place; tests must
# never rewrite or chmod it at runtime (a live writer makes exec fail ETXTBSY).
printf '%s\n' "$@" > "${FAKE_TUI_ARGS:?FAKE_TUI_ARGS is not set}"
