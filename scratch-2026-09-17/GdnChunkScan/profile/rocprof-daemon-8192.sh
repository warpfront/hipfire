#!/bin/sh
exec rocprofv3 --kernel-trace --stats --output-format csv --output-directory /home/kaden/hipfire-gdn-chunk-scan/scratch-2026-09-17/GdnChunkScan/profile/pp8192 --output-file trace -- /home/kaden/hipfire-gdn-chunk-scan/target/release/daemon "$@"
