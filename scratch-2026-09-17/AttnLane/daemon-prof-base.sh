#!/bin/sh
exec /usr/bin/python3 /home/kaden/ClaudeCode/warpfront/wt-raster2/scratch-2026-09-17/Raster2/bin/det-launch.py /home/kaden/ClaudeCode/warpfront/wt-raster2/scratch-2026-09-17/Raster2/bin/minpy-rocprof.py /home/kaden/ClaudeCode/warpfront/wt-attnlane/scratch-2026-09-17/AttnLane/rocprof-base trace-base /home/kaden/ClaudeCode/warpfront/wt-lloyd/target/release/daemon "$@"
