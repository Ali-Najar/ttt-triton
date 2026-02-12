import triton
import triton.language as tl
from kernels.utils import gelu_tanh, gelu_bwd


@triton.jit
def ttt_mlp_scan_forward(
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W1_init_ptr,
    b1_init_ptr,
    W2_init_ptr,
    b2_init_ptr,
    XQ_batch_ptr,
    XV_batch_ptr,
    XK_batch_ptr,
    eta_batch_ptr,
    W1_last_ptr,
    b1_last_ptr,
    W2_last_ptr,
    b2_last_ptr,
    XQW_batch_ptr,
    W1_checkpoints_ptr,
    b1_checkpoints_ptr,
    W2_checkpoints_ptr,
    b2_checkpoints_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    CS_FF_stride: tl.constexpr,
    F_FF_stride: tl.constexpr,
    FF_F_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    FF_stride: tl.constexpr,
    # Constexpr
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    FF: tl.constexpr,
    K: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
):
    # print("hi")
    batch = tl.program_id(0)
    head = tl.program_id(1)
    mp_dtype = XQ_batch_ptr.type.element_ty

    # base strides for checkpoint tensors
    K_F_FF_stride = K * F_FF_stride
    K_FF_stride = K * FF_stride
    K_FF_F_stride = K * FF_F_stride
    K_F_stride = K * F_stride

    # Offsets for current states
    W1_off = batch * NH * F_FF_stride + head * F_FF_stride + tl.arange(0, F)[:, None] * FF + tl.arange(0, FF)[None, :]
    b1_off = batch * NH * FF_stride + head * FF_stride + tl.arange(0, FF)[None, :]
    W2_off = batch * NH * FF_F_stride + head * FF_F_stride + tl.arange(0, FF)[:, None] * F + tl.arange(0, F)[None, :]
    b2_off = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    norm_off = head * F_stride + tl.arange(0, F)

    W1 = tl.load(W1_init_ptr + W1_off).to(tl.float32)
    b1 = tl.load(b1_init_ptr + b1_off).to(tl.float32)
    W2 = tl.load(W2_init_ptr + W2_off).to(tl.float32)
    b2 = tl.load(b2_init_ptr + b2_off).to(tl.float32)

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_off).to(tl.float32)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_off).to(tl.float32)[None, :]

    for i in range(NC):
        if i % checkpoint_group_size == 0:
            curr_k = i // checkpoint_group_size

            W1_ckpt_off = (
                batch * NH * K_F_FF_stride
                + head * K_F_FF_stride
                + curr_k * F_FF_stride
                + tl.arange(0, F)[:, None] * FF
                + tl.arange(0, FF)[None, :]
            )
            b1_ckpt_off = (
                batch * NH * K_FF_stride
                + head * K_FF_stride
                + curr_k * FF_stride
                + tl.arange(0, FF)[None, :]
            )
            W2_ckpt_off = (
                batch * NH * K_FF_F_stride
                + head * K_FF_F_stride
                + curr_k * FF_F_stride
                + tl.arange(0, FF)[:, None] * F
                + tl.arange(0, F)[None, :]
            )
            b2_ckpt_off = (
                batch * NH * K_F_stride
                + head * K_F_stride
                + curr_k * F_stride
                + tl.arange(0, F)[None, :]
            )

            tl.store(W1_checkpoints_ptr + W1_ckpt_off, W1)
            tl.store(b1_checkpoints_ptr + b1_ckpt_off, b1)
            tl.store(W2_checkpoints_ptr + W2_ckpt_off, W2)
            tl.store(b2_checkpoints_ptr + b2_ckpt_off, b2)

        CS_F_off = (
            batch * NH * NC * CS_F_stride
            + head * NC * CS_F_stride
            + i * CS_F_stride
            + tl.arange(0, CS)[:, None] * F
            + tl.arange(0, F)[None, :]
        )
        CS_CS_off = (
            batch * NH * NC * CS_CS_stride
            + head * NC * CS_CS_stride
            + i * CS_CS_stride
            + tl.arange(0, CS)[:, None] * CS
            + tl.arange(0, CS)[None, :]
        )
        last_CS_off = (
            batch * NH * NC * CS_CS_stride
            + head * NC * CS_CS_stride
            + i * CS_CS_stride
            + (CS - 1) * CS
            + tl.arange(0, CS)[:, None]
        )

        XQ = tl.load(XQ_batch_ptr + CS_F_off).to(tl.float32)
        XK = tl.load(XK_batch_ptr + CS_F_off).to(tl.float32)
        XV = tl.load(XV_batch_ptr + CS_F_off).to(tl.float32)
        last_eta = tl.load(eta_batch_ptr + last_CS_off).to(tl.float32)  # (CS,1)

        # --- MLP forward ---
        Z1 = tl.dot(XK.to(mp_dtype), W1.to(mp_dtype)) + b1              # (CS,FF)
        X2 = gelu_tanh(Z1)                                             # (CS,FF)
        Z2 = tl.dot(X2.to(mp_dtype), W2.to(mp_dtype)) + b2             # (CS,F)
        target = XV - XK                                               # (CS,F)

        # LN-fused-L2 bwd to get grads for update
        mu = (tl.sum(Z2, axis=1) / F)[:, None]
        var = (tl.sum((Z2 - mu) * (Z2 - mu), axis=1) / F)[:, None]
        std = tl.sqrt(var + 1e-6)
        x_hat = (Z2 - mu) / std

        y = ln_weight * x_hat + ln_bias
        grad_out = y - target
        grad_xhat = grad_out * ln_weight

        grad_Z2 = (
            (1.0 / F)
            * (
                F * grad_xhat
                - tl.sum(grad_xhat, axis=1)[:, None]
                - x_hat * tl.sum(grad_xhat * x_hat, axis=1)[:, None]
            )
            / std
        )
        grad_Z1 = tl.dot(grad_Z2.to(mp_dtype), tl.trans(W2).to(mp_dtype)) * gelu_bwd(Z1)  # (CS,FF)

        # updates
        W2 = W2 - tl.dot(tl.trans(last_eta * X2).to(mp_dtype), grad_Z2.to(mp_dtype))
        b2 = b2 - tl.sum(last_eta * grad_Z2, axis=0)[None, :]
        W1 = W1 - tl.dot(tl.trans(last_eta * XK).to(mp_dtype), grad_Z1.to(mp_dtype))
        b1 = b1 - tl.sum(last_eta * grad_Z1, axis=0)[None, :]

        # apply updated params on XQ
        Z1b = tl.dot(XQ.to(mp_dtype), W1.to(mp_dtype)) + b1
        X2b = gelu_tanh(Z1b)
        Z2b = tl.dot(X2b.to(mp_dtype), W2.to(mp_dtype)) + b2

        mu2 = (tl.sum(Z2b, axis=1) / F)[:, None]
        var2 = (tl.sum((Z2b - mu2) * (Z2b - mu2), axis=1) / F)[:, None]
        std2 = tl.sqrt(var2 + 1e-6)
        x_hat2 = (Z2b - mu2) / std2
        Z2b_ln = ln_weight * x_hat2 + ln_bias

        XQW = XQ + Z2b_ln
        tl.store(XQW_batch_ptr + CS_F_off, XQW)

    # store final states
    W1_last_off = W1_off
    b1_last_off = b1_off
    W2_last_off = W2_off
    b2_last_off = b2_off

    tl.store(W1_last_ptr + W1_last_off, W1)
    tl.store(b1_last_ptr + b1_last_off, b1)
    tl.store(W2_last_ptr + W2_last_off, W2)
    tl.store(b2_last_ptr + b2_last_off, b2)