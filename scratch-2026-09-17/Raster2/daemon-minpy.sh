#!/bin/sh
exec /usr/bin/python3 -c "import sys,subprocess; p=subprocess.Popen(sys.argv[1:],stdin=sys.stdin.buffer,stdout=sys.stdout.buffer); sys.exit(p.wait())" /home/kaden/ClaudeCode/warpfront/wt-raster2/scratch-2026-09-17/Raster2/bin/daemon-base "$@"
