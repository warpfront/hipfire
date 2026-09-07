# SPDX-License-Identifier: MIT OR Apache-2.0
# T0 gate-(c) CPU baseline: PIL decode + BICUBIC resize to smart_resize dims +
# normalize + patchify. Conservative proxy for crates/hipfire-arch-qwen35-vl
# image.rs load_and_preprocess + extract_patches (real path uses the `image`
# crate's slower pure-Rust JPEG decoder, so beating this proxy beats the repo
# path a fortiori). Fresh process per invocation; caller runs >=3x.
import sys, time
import numpy as np
from PIL import Image

FACTOR = 32          # patch_size(16) * sms(2), qwen35-vl
MIN_PX, MAX_PX = 65536, 2_000_000

def smart_resize(h, w):
    scale = 1.0
    if h * w > MAX_PX:
        scale = (MAX_PX / (h * w)) ** 0.5
    elif h * w < MIN_PX:
        scale = (MIN_PX / (h * w)) ** 0.5
    nh, nw = round(h * scale), round(w * scale)
    nh = max(FACTOR, (nh // FACTOR) * FACTOR)
    nw = max(FACTOR, (nw // FACTOR) * FACTOR)
    return nh, nw

def run(path, reps):
    raw = open(path, 'rb').read()
    import io
    # cold: full pipeline once
    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    ow, oh = img.size
    nh, nw = smart_resize(oh, ow)
    img = img.resize((nw, nh), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # HWC
    chw = np.transpose(a, (2, 0, 1))                        # CHW
    # extract_patches analogue: 16px patches, 2x2-grouped
    P, SMS = 16, 2
    gh, gw = nh // P, nw // P
    x = chw.reshape(3, gh, P, gw, P).transpose(1, 3, 0, 2, 4)
    x = x.reshape(gh // SMS, SMS, gw // SMS, SMS, 3, P, P)
    x = x.transpose(0, 2, 1, 3, 4, 5, 6).reshape(-1)
    cold = (time.perf_counter() - t0) * 1e3
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        img = img.resize((nw, nh), Image.BICUBIC)
        a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        chw = np.transpose(a, (2, 0, 1))
        x = chw.reshape(3, gh, P, gw, P).transpose(1, 3, 0, 2, 4)
        x = x.reshape(gh // SMS, SMS, gw // SMS, SMS, 3, P, P)
        x = x.transpose(0, 2, 1, 3, 4, 5, 6).reshape(-1)
        dt = (time.perf_counter() - t0) * 1e3
        best = min(best, dt)
    print(f'cpu {path} {ow}x{oh}->{nw}x{nh} patches={x.size//(3*P*P)} cold={cold:.2f}ms warm_min={best:.2f}ms')

if __name__ == '__main__':
    path, reps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run(path, reps)
