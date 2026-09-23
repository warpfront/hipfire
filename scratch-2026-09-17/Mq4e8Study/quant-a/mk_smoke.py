#!/usr/bin/env python3
"""Build a tiny fake qwen3_5 safetensors dir for mq4e8 routing smoke tests."""
import json
import os

import numpy as np
from safetensors.numpy import save_file

D = "/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/smoke_tiny"
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
w = (rng.standard_normal((4, 256)) * 0.05).astype(np.float32)
# proper bf16: take float32 bits >> 16, store as uint16 raw
raw = w.view(np.uint32)
bf16raw = (raw >> 16).astype(np.uint16)
import struct

payload = bf16raw.tobytes()
# minimal safetensors: header {"name": {dtype, shape, offsets}} + data
header = {
    "model.language_model.layers.0.mlp.gate_proj.weight": {
        "dtype": "BF16",
        "shape": [4, 256],
        "data_offsets": [0, len(payload)],
    }
}
hjson = json.dumps(header).encode()
with open(os.path.join(D, "model-00001-of-00001.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hjson)))
    f.write(hjson)
    f.write(payload)
with open(os.path.join(D, "config.json"), "w") as f:
    json.dump({"model_type": "qwen3_5", "hidden_size": 256}, f)
print("wrote", D)
