import torch
import triton
import triton.language as tl

from kernels.utils import gelu_tanh

@triton.jit
def mlp_cached_readout_kernel(
    # inputs
    XQ_ptr,      # [B, NH, NCseg, CS, F]
    W1_ptr,      # [B, NH, F, FF]
    b1_ptr,      # [B, NH, 1, FF]
    W2_ptr,      # [B, NH, FF, F]
    b2_ptr,      # [B, NH, 1, F]
    ln_w_ptr,    # [NH, F]
    ln_b_ptr,    # [NH, F]
    # output
    OUT_ptr,     # [B, NH, NCseg, CS, F]

    # strides (element strides)
    CS_F_stride: tl.constexpr,   # CS*F
    F_FF_stride: tl.constexpr,   # F*FF
    FF_F_stride: tl.constexpr,   # FF*F
    F_stride: tl.constexpr,      # F
    FF_stride: tl.constexpr,     # FF

    # constants
    NH: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    FF: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    nc = tl.program_id(2)  # within segment

    mp_dtype = XQ_ptr.type.element_ty

    # ---- offsets ----
    # XQ mini-batch base
    xq_off = (
        b * NH * 0  # batch stride handled by base pointer arithmetic from torch contiguous
    )

    # In contiguous [B, NH, NCseg, CS, F], the flattened offset is:
    # (((b*NH + h)*NCseg + nc)*CS + cs)*F + f
    # we don’t need NCseg explicitly; torch slice makes it contiguous.

    xq_base = ((b * NH + h) * 0)  # dummy; we’ll compute with element offsets below

    # load ln params for this head
    norm_off = h * F_stride + tl.arange(0, F)
    ln_w = tl.load(ln_w_ptr + norm_off).to(tl.float32)[None, :]   # (1,F)
    ln_b = tl.load(ln_b_ptr + norm_off).to(tl.float32)[None, :]   # (1,F)

    # load XQ (CS,F)
    cs = tl.arange(0, CS)[:, None]
    f = tl.arange(0, F)[None, :]

    # base pointer offsets for XQ/OUT:
    xq_off = (
        (b * NH * 0)  # unused
    )

    # use element offsets with known contiguous layout:
    XQ_off = (
        (((b * NH + h) * 0 + nc) * CS_F_stride)  # (b,h,nc) block
        + cs * F
        + f
    )
    XQ = tl.load(XQ_ptr + XQ_off).to(tl.float32)  # (CS,F)

    # load weights/bias (for this b,h)
    # W1: [F,FF]
    W1_off = (b * NH + h) * F_FF_stride + tl.arange(0, F)[:, None] * FF + tl.arange(0, FF)[None, :]
    W1 = tl.load(W1_ptr + W1_off).to(tl.float32)

    b1_off = (b * NH + h) * FF_stride + tl.arange(0, FF)[None, :]
    b1 = tl.load(b1_ptr + b1_off).to(tl.float32)  # (1,FF)

    # W2: [FF,F]
    W2_off = (b * NH + h) * FF_F_stride + tl.arange(0, FF)[:, None] * F + tl.arange(0, F)[None, :]
    W2 = tl.load(W2_ptr + W2_off).to(tl.float32)

    b2_off = (b * NH + h) * F_stride + tl.arange(0, F)[None, :]
    b2 = tl.load(b2_ptr + b2_off).to(tl.float32)  # (1,F)

    # ---- MLP readout ----
    # Z1 = XQ @ W1 + b1   => (CS,FF)
    Z1 = tl.dot(XQ.to(mp_dtype), W1.to(mp_dtype)).to(tl.float32) + b1
    X2 = gelu_tanh(Z1)  # (CS,FF)

    # Z2 = X2 @ W2 + b2   => (CS,F)
    Z2 = tl.dot(X2.to(mp_dtype), W2.to(mp_dtype)).to(tl.float32) + b2

    # ---- LN over last dim F ----
    mu = tl.sum(Z2, axis=1)[:, None] / F
    var = tl.sum((Z2 - mu) * (Z2 - mu), axis=1)[:, None] / F
    std = tl.sqrt(var + 1e-6)
    xhat = (Z2 - mu) / std
    Z2_ln = ln_w * xhat + ln_b   # (CS,F)

    # store
    OUT_off = (
        (((b * NH + h) * 0 + nc) * CS_F_stride)
        + cs * F
        + f
    )
    tl.store(OUT_ptr + OUT_off, Z2_ln.to(OUT_ptr.type.element_ty))


def mlp_cached_readout_triton(XQ_seg, W1, b1, W2, b2, ln_w, ln_b):
    """
    XQ_seg: [B, NH, NCseg, CS, F] (contiguous)
    W1:     [B, NH, F, FF]
    b1:     [B, NH, 1, FF]
    W2:     [B, NH, FF, F]
    b2:     [B, NH, 1, F]
    ln_w/b: [NH, F]
    returns: [B, NH, NCseg, CS, F] (float32)
    """
    assert XQ_seg.is_cuda and W1.is_cuda
    B, NH, NCseg, CS, F = XQ_seg.shape
    FF = W1.shape[-1]

    out = torch.empty((B, NH, NCseg, CS, F), device=XQ_seg.device, dtype=torch.float32)

    grid = (B, NH, NCseg)
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
        num_warps=4,
    )
    return XQ_seg + out
