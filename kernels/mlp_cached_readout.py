import torch
import triton
import triton.language as tl
import math
from kernels.utils import gelu_tanh

@triton.jit
def mlp_cached_readout_kernel(
    # inputs
    XQ_ptr,      # [B, NH, NCseg, CS, F]
    W1_ptr,      # [B, NH, M, F, FF]
    b1_ptr,      # [B, NH, M, 1, FF]
    W2_ptr,      # [B, NH, M, FF, F]
    b2_ptr,      # [B, NH, M, 1, F]
    ln_w_ptr,    # [NH, F]
    ln_b_ptr,    # [NH, F]
    # output
    OUT_ptr,     # [B, NH, NCseg, M, CS, F]

    # strides
    CS_F_stride: tl.constexpr,
    F_FF_stride: tl.constexpr,
    FF_F_stride: tl.constexpr,
    F_stride: tl.constexpr,
    FF_stride: tl.constexpr,

    # constants
    NH: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    FF: tl.constexpr,
    M: tl.constexpr,
    NCseg: tl.constexpr,
):
    # Flatten B and NH into pid_0 since Triton limits grids to 3D
    pid_0 = tl.program_id(0)
    b = pid_0 // NH
    h = pid_0 % NH
    nc = tl.program_id(1)
    m = tl.program_id(2)

    mp_dtype = XQ_ptr.type.element_ty

    # load ln params
    norm_off = h * F_stride + tl.arange(0, F)
    ln_w = tl.load(ln_w_ptr + norm_off).to(tl.float32)[None, :]
    ln_b = tl.load(ln_b_ptr + norm_off).to(tl.float32)[None, :]

    # load XQ
    cs = tl.arange(0, CS)[:, None]
    f = tl.arange(0, F)[None, :]
    XQ_off = (((b * NH + h) * NCseg + nc) * CS_F_stride) + cs * F + f
    XQ = tl.load(XQ_ptr + XQ_off).to(tl.float32)

    # load weights/bias mapped to the specific past segment (m)
    W1_base = ((b * NH + h) * M + m) * F_FF_stride
    W1_off = W1_base + tl.arange(0, F)[:, None] * FF + tl.arange(0, FF)[None, :]
    W1 = tl.load(W1_ptr + W1_off).to(tl.float32)

    b1_base = ((b * NH + h) * M + m) * FF_stride
    b1_off = b1_base + tl.arange(0, FF)[None, :]
    b1 = tl.load(b1_ptr + b1_off).to(tl.float32)

    W2_base = ((b * NH + h) * M + m) * FF_F_stride
    W2_off = W2_base + tl.arange(0, FF)[:, None] * F + tl.arange(0, F)[None, :]
    W2 = tl.load(W2_ptr + W2_off).to(tl.float32)

    b2_base = ((b * NH + h) * M + m) * F_stride
    b2_off = b2_base + tl.arange(0, F)[None, :]
    b2 = tl.load(b2_ptr + b2_off).to(tl.float32)

    # MLP readout
    Z1 = tl.dot(XQ.to(mp_dtype), W1.to(mp_dtype)).to(tl.float32) + b1
    X2 = gelu_tanh(Z1)
    Z2 = tl.dot(X2.to(mp_dtype), W2.to(mp_dtype)).to(tl.float32) + b2

    # LN
    mu = tl.sum(Z2, axis=1)[:, None] / F
    var = tl.sum((Z2 - mu) * (Z2 - mu), axis=1)[:, None] / F
    std = tl.sqrt(var + 1e-6)
    xhat = (Z2 - mu) / std
    Z2_ln = ln_w * xhat + ln_b

    # store
    OUT_base = (((b * NH + h) * NCseg + nc) * M + m) * CS_F_stride
    OUT_off = OUT_base + cs * F + f
    tl.store(OUT_ptr + OUT_off, Z2_ln.to(OUT_ptr.type.element_ty))


def mlp_cached_readout_triton(XQ_seg, W1, b1, W2, b2, ln_w, ln_b):
    assert XQ_seg.is_cuda and W1.is_cuda
    B, NH, M, F, FF = W1.shape
    _, _, NCseg, CS, _ = XQ_seg.shape

    out = torch.empty((B, NH, NCseg, M, CS, F), device=XQ_seg.device, dtype=torch.float32)

    grid = (B * NH, NCseg, M)
    mlp_cached_readout_kernel[grid](
        XQ_seg, W1, b1, W2, b2, ln_w, ln_b, out,
        CS_F_stride=CS * F,
        F_FF_stride=F * FF,
        FF_F_stride=FF * F,
        F_stride=F,
        FF_stride=FF,
        NH=NH,
        CS=CS,
        F=F,
        FF=FF,
        M=M,
        NCseg=XQ_seg.shape[2],
        num_warps=4,
    )
    return out