#!/usr/bin/env python3
# Reference conv2d downsampler forward for LFM3-Audio, byte-matching the HF
# `Lfm3AudioDownsampler.forward` (modeling_lfm3_audio.py). Loads the actual
# downsampler weights from the LFM3-Audio-2B safetensors (BF16) and runs the
# forward on the mel reference produced by gen_mel_reference.py.
#
# Outputs (raw f32) into tests/fixtures/:
#   downsample_out.bin  [n_tokens * 768] row-major (n_tokens, 768) — encoder input
#   downsample_meta.txt n_tokens / dims
import json, struct
import numpy as np
from pathlib import Path

MODEL = "/data/liquid-ai/LFM3-Audio-2B/model.safetensors"
FIX = Path(__file__).parent

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    # map name -> (offset, dtype, shape) + the file handle for lazy reads
    return tensors, path

def read_tensor(tensors, path, name):
    info = tensors[name]
    dtype = info["dtype"]
    shape = [int(s) for s in info["shape"]]
    offset = int(info["data_offsets"][0])
    n_bytes = int(info["data_offsets"][1]) - offset
    with open(path, "rb") as f:
        f.seek(offset + 8 + len(json.dumps({k:v for k,v in tensors.items()}).encode()))  # not exact; use header len
    # recompute header length properly
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        f.read(hlen)  # skip header
        f.seek(offset)
        raw = f.read(n_bytes)
    numel = 1
    for s in shape:
        numel *= s
    if dtype == "BF16":
        a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        a = a + (a << 16)  # BF16 → F32: f32 = bf16 bits in high 16, low 0
        arr = np.frombuffer(a.tobytes(), dtype=np.float32).copy()
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).copy()
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    return arr.reshape(shape)

def silu(x):
    return x / (1.0 + np.exp(-x))

def conv2d_k3s2p1(x, w, b):
    # x: [B, Cin, H, W], w: [Cout, Cin, 3, 3], b: [Cout]
    B, Cin, H, W = x.shape
    Cout = w.shape[0]
    Ho = (H - 1) // 2 + 1
    Wo = (W - 1) // 2 + 1
    xp = np.zeros((B, Cin, H + 2, W + 2), dtype=x.dtype)
    xp[:, :, 1:1+H, 1:1+W] = x
    out = np.zeros((B, Cout, Ho, Wo), dtype=np.float32)
    for co in range(Cout):
        wc = w[co]  # [Cin, 3, 3]
        for ho in range(Ho):
            for wo in range(Wo):
                win = xp[:, :, ho*2:ho*2+3, wo*2:wo*2+3]  # [B, Cin, 3, 3]
                out[:, co, ho, wo] = b[co] + np.sum(win * wc, axis=(1, 2, 3))
    return out

def conv2d_dw_k3s2p1(x, w, b):
    # x: [B, C, H, W], w: [C, 1, 3, 3], b: [C]
    B, C, H, W = x.shape
    Ho = (H - 1) // 2 + 1
    Wo = (W - 1) // 2 + 1
    xp = np.zeros((B, C, H + 2, W + 2), dtype=x.dtype)
    xp[:, :, 1:1+H, 1:1+W] = x
    out = np.zeros((B, C, Ho, Wo), dtype=np.float32)
    for c in range(C):
        wc = w[c, 0]  # [3, 3]
        for ho in range(Ho):
            for wo in range(Wo):
                win = xp[:, c, ho*2:ho*2+3, wo*2:wo*2+3]  # [B, 3, 3]
                out[:, c, ho, wo] = b[c] + np.sum(win * wc, axis=(1, 2))
    return out

def conv2d_1x1(x, w, b):
    # x: [B, Cin, H, W], w: [Cout, Cin, 1, 1], b: [Cout]
    B, Cin, H, W = x.shape
    Cout = w.shape[0]
    out = np.einsum("bchw,oc->bohw", x, w[:, :, 0, 0]) + b[None, :, None, None]
    return out.astype(np.float32)

def conv1d_k1(x, w, b):
    # x: [B, Cin, L], w: [Cout, Cin, 1], b: [Cout]
    out = np.einsum("bcl,oc->bol", x, w[:, :, 0]) + b[None, :, None]
    return out.astype(np.float32)

def main():
    tensors, path = load_safetensors(MODEL)
    P = "encoder.downsample."
    conv_in_w = read_tensor(tensors, path, P + "conv_in.weight")      # [256,1,3,3]
    conv_in_b = read_tensor(tensors, path, P + "conv_in.bias")        # [256]
    conv_mid_w = read_tensor(tensors, path, P + "conv_mid.weight")    # [256,1,3,3]
    conv_mid_b = read_tensor(tensors, path, P + "conv_mid.bias")
    lin_mid_w = read_tensor(tensors, path, P + "lin_mid.weight")      # [256,256,1,1]
    lin_mid_b = read_tensor(tensors, path, P + "lin_mid.bias")
    conv_end_w = read_tensor(tensors, path, P + "conv_end.weight")    # [256,1,3,3]
    conv_end_b = read_tensor(tensors, path, P + "conv_end.bias")
    lin_end_w = read_tensor(tensors, path, P + "lin_end.weight")      # [256,256,1,1]
    lin_end_b = read_tensor(tensors, path, P + "lin_end.bias")
    lin_out_w = read_tensor(tensors, path, P + "lin_out.weight")      # [768,4096,1]
    lin_out_b = read_tensor(tensors, path, P + "lin_out.bias")        # [768]

    # mel reference: [128, T] (from gen_mel_reference.py)
    mel = np.fromfile(FIX / "log_mel.bin", dtype=np.float32)
    with open(FIX / "meta.txt") as f:
        meta = dict(l.split("=") for l in f.read().splitlines() if "=" in l)
    n_mels = int(meta["n_mels"])
    num_frames = int(meta["num_frames"])
    mel = mel.reshape(n_mels, num_frames)
    print(f"mel input: {mel.shape} ({num_frames} frames)")

    # forward (B=1)
    h = mel[None, None, :, :]  # [1, 1, 128, T]
    h = silu(conv2d_k3s2p1(h, conv_in_w, conv_in_b))                 # [1, 256, 64, T2]
    h = silu(conv2d_1x1(conv2d_dw_k3s2p1(h, conv_mid_w, conv_mid_b), lin_mid_w, lin_mid_b))  # [1, 256, 32, T4]
    h = silu(conv2d_1x1(conv2d_dw_k3s2p1(h, conv_end_w, conv_end_b), lin_end_w, lin_end_b))  # [1, 256, 16, T8]
    h = h.reshape(1, h.shape[1] * h.shape[2], h.shape[3])  # flatten(1,2) → [1, 4096, T8]
    out = conv1d_k1(h, lin_out_w, lin_out_b)               # [1, 768, T8]
    out = out[0].T.astype(np.float32)                      # [T8, 768]
    print(f"downsample out: {out.shape} ({out.shape[0]} tokens)")

    out.tofile(FIX / "downsample_out.bin")
    with open(FIX / "downsample_meta.txt", "w") as f:
        f.write(f"n_tokens={out.shape[0]}\n")
        f.write(f"hidden={out.shape[1]}\n")

    # Also dump each downsampler weight as F32 for the Rust unit test.
    weights = {
        "conv_in.weight": conv_in_w, "conv_in.bias": conv_in_b,
        "conv_mid.weight": conv_mid_w, "conv_mid.bias": conv_mid_b,
        "lin_mid.weight": lin_mid_w, "lin_mid.bias": lin_mid_b,
        "conv_end.weight": conv_end_w, "conv_end.bias": conv_end_b,
        "lin_end.weight": lin_end_w, "lin_end.bias": lin_end_b,
        "lin_out.weight": lin_out_w, "lin_out.bias": lin_out_b,
    }
    import json as _json
    idx = {}
    for name, arr in weights.items():
        a = np.ascontiguousarray(arr.astype(np.float32).reshape(-1))
        a.tofile(FIX / f"dw_{name.replace('.', '_')}.bin")
        idx[name] = {"shape": list(arr.shape), "numel": int(a.size)}
    with open(FIX / "downsample_weights_idx.json", "w") as f:
        _json.dump(idx, f, indent=2)
    print("dumped downsampler weights (F32)")

main()
