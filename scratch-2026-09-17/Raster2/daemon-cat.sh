#!/bin/sh
exec /usr/bin/python3 /home/kaden/ClaudeCode/warpfront/wt-raster2/scratch-2026-09-17/Raster2/bin/dbg-relay.py /home/kaden/ClaudeCode/warpfront/wt-raster2/scratch-2026-09-17/Raster2/dbg-cat.log /tmp/nonexistent trace-cat /bin/cat "$@"
