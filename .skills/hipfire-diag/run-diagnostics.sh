#!/bin/bash
# hipfire GPU diagnostics — outputs structured JSON for any agent to interpret.
# Usage: .skills/hipfire-diag/run-diagnostics.sh [model.hfq]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

echo '{'
echo '  "tool": "hipfire-diag",'
echo '  "version": "0.0.1",'

# 1. GPU detection
echo '  "gpu": {'
if [ -e /dev/kfd ]; then
    echo '    "kfd": true,'
    ARCH=$(rocm-smi --showproductname 2>/dev/null | grep -i "card" | head -1 || echo "unknown")
    VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Memory" | grep -oP '\d+' | head -1 || echo "0")
    echo "    \"card\": \"$ARCH\","
    echo "    \"vram_bytes\": $VRAM,"
    # Try to detect arch via hipcc or our binary
    if [ -f target/release/examples/test_kernels ]; then
        GPU_ARCH=$(timeout 5 target/release/examples/test_kernels 2>&1 | grep "GPU:" | head -1 | sed 's/.*GPU: //' | cut -d' ' -f1 || echo "unknown")
        echo "    \"arch\": \"$GPU_ARCH\""
    else
        echo '    "arch": "unknown (build test_kernels first)"'
    fi
else
    echo '    "kfd": false,'
    echo '    "error": "No AMD GPU detected (/dev/kfd missing). Install amdgpu driver."'
fi
echo '  },'

# 2. Pre-compiled kernels
echo '  "kernels": {'
for arch in gfx1010 gfx1030 gfx1100 gfx1200 gfx1201; do
    hsaco=$(ls kernels/compiled/$arch/*.hsaco 2>/dev/null | wc -l)
    hashes=$(ls kernels/compiled/$arch/*.hash 2>/dev/null | wc -l)
    echo "    \"$arch\": {\"blobs\": $hsaco, \"hashes\": $hashes},"
done
echo '    "_note": "blobs=pre-compiled kernels, hashes=integrity sidecar files"'
echo '  },'

# 3. Kernel tests (no model needed)
echo '  "kernel_tests": {'
if [ -f target/release/examples/test_kernels ]; then
    RESULT=$(timeout 60 target/release/examples/test_kernels 2>&1)
    PASSED=$(echo "$RESULT" | grep "Passed:" | grep -oP '\d+' || echo "0")
    FAILED=$(echo "$RESULT" | grep "Failed:" | grep -oP '\d+' || echo "0")
    echo "    \"passed\": $PASSED,"
    echo "    \"failed\": $FAILED,"
    if [ "$FAILED" != "0" ]; then
        FAILURES=$(echo "$RESULT" | grep "FAIL\|PANIC" | head -5 | sed 's/"/\\"/g' | while read -r line; do echo "      \"$line\","; done)
        echo "    \"failures\": ["
        echo "$FAILURES"
        echo '      "_end": true'
        echo "    ]"
    else
        echo '    "failures": []'
    fi
elif [ -f target/release/examples/test_kernelsQA ]; then
    RESULT=$(timeout 60 target/release/examples/test_kernelsQA 2>&1)
    PASSED=$(echo "$RESULT" | grep "Passed:" | grep -oP '\d+' || echo "0")
    FAILED=$(echo "$RESULT" | grep "Failed:" | grep -oP '\d+' || echo "0")
    echo "    \"passed\": $PASSED,"
    echo "    \"failed\": $FAILED"
else
    echo '    "error": "No test binary found. Run: cargo build --release --features deltanet --example test_kernels -p hipfire-runtime"'
fi
echo '  },'

# 4. Inference tests (if model provided)
MODEL="${1:-}"
echo '  "inference_tests": {'
if [ -n "$MODEL" ] && [ -f "$MODEL" ]; then
    if [ -f target/release/examples/test_inference ]; then
        RESULT=$(timeout 120 target/release/examples/test_inference "$MODEL" 2>&1)
        PASSED=$(echo "$RESULT" | grep "Passed:" | grep -oP '\d+' || echo "0")
        FAILED=$(echo "$RESULT" | grep "Failed:" | grep -oP '\d+' || echo "0")
        SPEED=$(echo "$RESULT" | grep "tok/s" | tail -1 | grep -oP '[\d.]+(?= tok/s)' | tail -1 || echo "0")
        LEAK=$(echo "$RESULT" | grep "LEAK\|leak" | head -1 || echo "none")
        echo "    \"model\": \"$MODEL\","
        echo "    \"passed\": $PASSED,"
        echo "    \"failed\": $FAILED,"
        echo "    \"tok_s\": $SPEED,"
        echo "    \"vram_leak\": \"$LEAK\""
    else
        echo '    "error": "No inference test binary. Run: cargo build --release --features deltanet --example test_inference -p hipfire-runtime"'
    fi
elif [ -n "$MODEL" ]; then
    echo "    \"error\": \"Model not found: $MODEL\""
else
    echo '    "skipped": "No model provided. Pass model.hfq as argument for inference tests."'
fi
echo '  },'

# 5. Build status
echo '  "build": {'
if [ -f target/release/examples/infer ]; then
    echo '    "infer": true,'
else
    echo '    "infer": false,'
fi
if [ -f target/release/examples/infer_hfq ]; then
    echo '    "infer_hfq": true'
else
    echo '    "infer_hfq": false'
fi
echo '  }'

echo '}'
