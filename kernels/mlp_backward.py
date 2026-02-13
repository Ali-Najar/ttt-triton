import triton
import triton.language as tl
from kernels.utils import gelu_tanh, gelu_bwd, gelu_bwd_derivative


@triton.jit
def ttt_mlp_mini_batch_forward(
    W1_init,
    b1_init,
    W2_init,
    b2_init,
    ln_weight,
    ln_bias,
    XQ,
    XK,
    XV,
    last_eta,
    CS: tl.constexpr,
    F: tl.constexpr,
    FF: tl.constexpr,
    mp_dtype,
):
    Z1 = tl.dot(XK.to(mp_dtype), W1_init.to(mp_dtype)) + b1_init            # (CS,FF)
    X2 = gelu_tanh(Z1)                                                     # (CS,FF)
    Z2 = tl.dot(X2.to(mp_dtype), W2_init.to(mp_dtype)) + b2_init           # (CS,F)
    target = XV - XK                                                       # (CS,F)

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
    grad_Z1 = tl.dot(grad_Z2.to(mp_dtype), tl.trans(W2_init).to(mp_dtype)) * gelu_bwd(Z1)  # (CS,FF)

    W2_last = W2_init - tl.dot(tl.trans(last_eta * X2).to(mp_dtype), grad_Z2.to(mp_dtype))  # (FF,F)
    b2_last = b2_init - tl.sum(last_eta * grad_Z2, axis=0)[None, :]                         # (1,F)
    W1_last = W1_init - tl.dot(tl.trans(last_eta * XK).to(mp_dtype), grad_Z1.to(mp_dtype))  # (F,FF)
    b1_last = b1_init - tl.sum(last_eta * grad_Z1, axis=0)[None, :]                         # (1,FF)

    Z1b = tl.dot(XQ.to(mp_dtype), W1_last.to(mp_dtype)) + b1_last            # (CS,FF)
    X2b = gelu_tanh(Z1b)                                                     # (CS,FF)
    Z2b = tl.dot(X2b.to(mp_dtype), W2_last.to(mp_dtype)) + b2_last           # (CS,F)

    mu2 = (tl.sum(Z2b, axis=1) / F)[:, None]
    var2 = (tl.sum((Z2b - mu2) * (Z2b - mu2), axis=1) / F)[:, None]
    std2 = tl.sqrt(var2 + 1e-6)
    x_hat2 = (Z2b - mu2) / std2
    Z2b_ln = ln_weight * x_hat2 + ln_bias

    XQW = XQ + Z2b_ln

    return (
        XQW,
        W1_last,
        b1_last,
        W2_last,
        b2_last,
        Z1,
        Z1b,
        X2,
        X2b,
        x_hat2,
        std2,
        grad_Z1,
        grad_Z2,
        x_hat,
        grad_xhat,
        grad_out,
        std,
    )


@triton.jit
def ttt_mlp_mini_batch_backward(
    XQ,
    XK,
    W1_init,
    W1_last,
    W2_init,
    W2_last,
    Z1,
    Z1_bar,
    X2,
    X2_bar,
    ln_weight,
    std_fused,
    x_hat_fused,
    grad_output_fused,
    grad_x_hat_fused,
    grad_l_wrt_Z1,
    grad_l_wrt_Z2,
    last_eta,
    std_ln,
    x_hat_ln,
    grad_L_W1_last,
    grad_L_b1_last,
    grad_L_W2_last,
    grad_L_b2_last,
    grad_L_XQW,
    CS: tl.constexpr,
    F: tl.constexpr,
    FF: tl.constexpr,
    mp_dtype=None,
):
    # LN backward
    grad_L_ln_weight_ln = tl.sum(grad_L_XQW * x_hat_ln, axis=0)
    grad_L_ln_bias_ln = tl.sum(grad_L_XQW, axis=0)

    grad_L_xhat_ln = grad_L_XQW * ln_weight
    grad_L_Z2_bar = (
        (1.0 / F)
        * (
            F * grad_L_xhat_ln
            - tl.sum(grad_L_xhat_ln, axis=1)[:, None]
            - x_hat_ln * tl.sum(grad_L_xhat_ln * x_hat_ln, axis=1)[:, None]
        )
        / std_ln
    )

    grad_L_X2_bar = tl.dot(grad_L_Z2_bar.to(mp_dtype), tl.trans(W2_last).to(mp_dtype))      # (CS,FF)
    grad_L_Z1_bar = grad_L_X2_bar * gelu_bwd(Z1_bar)                                        # (CS,FF)

    grad_L_W2_last += tl.dot(tl.trans(X2_bar).to(mp_dtype), grad_L_Z2_bar.to(mp_dtype))     # (FF,F)
    grad_L_b2_last += tl.sum(grad_L_Z2_bar, axis=0)[None, :]                                # (1,F)
    grad_L_W1_last += tl.dot(tl.trans(XQ).to(mp_dtype), grad_L_Z1_bar.to(mp_dtype))         # (F,FF)
    grad_L_b1_last += tl.sum(grad_L_Z1_bar, axis=0)[None, :]                                # (1,FF)

    # grads wrt inner grads
    grad_L_grad_Z1 = -(
        tl.dot((last_eta * XK).to(mp_dtype), grad_L_W1_last.to(mp_dtype))
    ) - (last_eta * grad_L_b1_last)

    grad_L_grad_Z2 = -(
        tl.dot((last_eta * X2).to(mp_dtype), grad_L_W2_last.to(mp_dtype))
    ) - (last_eta * grad_L_b2_last) + tl.dot((grad_L_grad_Z1 * gelu_bwd(Z1)).to(mp_dtype), W2_init.to(mp_dtype))

    # XQ path
    grad_L_XQ_mini = tl.dot(grad_L_Z1_bar.to(mp_dtype), tl.trans(W1_last).to(mp_dtype))     # (CS,F)

    # dual-form pieces
    grad_Z1_last = tl.trans(tl.dot(grad_L_W1_last.to(mp_dtype), tl.trans(grad_l_wrt_Z1).to(mp_dtype)))  # (CS,F)
    grad_Z2_last = tl.trans(tl.dot(grad_L_W2_last.to(mp_dtype), tl.trans(grad_l_wrt_Z2).to(mp_dtype)))  # (CS,FF)

    grad_L_XK_mini = -grad_Z1_last * last_eta                                                # (CS,F)

    # Each term becomes (CS,) by summing over feature dim, then we pack to (1, CS)
    term_w2 = tl.sum(grad_Z2_last * X2, axis=1)                    # (CS,)  FF reduced
    term_b2 = tl.sum(grad_L_b2_last * grad_l_wrt_Z2, axis=1)       # (CS,)  F reduced
    term_w1 = tl.sum(grad_Z1_last * XK, axis=1)         # (CS,)  F reduced
    term_b1 = tl.sum(grad_L_b1_last * grad_l_wrt_Z1, axis=1)       # (CS,)  FF reduced

    grad_L_last_eta = -(term_w2 + term_b2 + term_w1 + term_b1)[None, :]  # (1, CS)

    last_row_mask = tl.arange(0, CS)[:, None] == (CS - 1)
    grad_L_eta = tl.where(last_row_mask, grad_L_last_eta, 0)

    # back through fused LN in forward (wrt W2_init etc)
    grad_L_W2_init_extra = tl.dot(tl.trans(grad_L_grad_Z1 * gelu_bwd(Z1)).to(mp_dtype), grad_l_wrt_Z2.to(mp_dtype))  # (FF,F)

    grad_L_grad_xhat_fused = (
        (1.0 / std_fused) * grad_L_grad_Z2
        + (1.0 / F) * tl.sum(-grad_L_grad_Z2 * (1.0 / std_fused), axis=1)[:, None]
        + (1.0 / F) * x_hat_fused * tl.sum(-grad_L_grad_Z2 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
    )

    grad_L_y = ln_weight * grad_L_grad_xhat_fused

    grad_L_ln_weight_fused = tl.sum(grad_output_fused * grad_L_grad_xhat_fused + grad_L_y * x_hat_fused, axis=0)
    grad_L_ln_bias_fused = tl.sum(grad_L_y, axis=0)

    grad_L_xhat_fused2 = (
        grad_L_y * ln_weight
        + (1.0 / F) * grad_x_hat_fused * tl.sum(-grad_L_grad_Z2 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
        + (1.0 / F) * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None] * (-grad_L_grad_Z2 * (1.0 / std_fused))
    )

    grad_L_std = -grad_L_xhat_fused2 * (x_hat_fused / std_fused) - (
        grad_L_grad_Z2 * (grad_l_wrt_Z2 / std_fused)
    )

    grad_L_Z2 = (
        grad_L_xhat_fused2 * (1.0 / std_fused)
        - (1.0 / F) * tl.sum(grad_L_xhat_fused2, axis=1)[:, None] * (1.0 / std_fused)
        + (1.0 / F) * tl.sum(grad_L_std, axis=1)[:, None] * x_hat_fused
    )

    grad_L_target = -ln_weight * grad_L_grad_xhat_fused   # (CS,F)

    # stage1 back
    grad_L_X2 = (
        tl.dot(grad_L_Z2.to(mp_dtype), tl.trans(W2_init).to(mp_dtype))
        - tl.dot(grad_l_wrt_Z2.to(mp_dtype), tl.trans(grad_L_W2_last).to(mp_dtype)) * last_eta
    )  # (CS,FF)

    grad_L_Z1 = grad_L_X2 * gelu_bwd(Z1) + (
        tl.dot(grad_l_wrt_Z2.to(mp_dtype), tl.trans(W2_init).to(mp_dtype))
    ) * grad_L_grad_Z1 * gelu_bwd_derivative(Z1)

    # grads to inputs
    grad_L_XQ = grad_L_XQW + grad_L_XQ_mini
    grad_L_XV = grad_L_target
    grad_L_XK = -grad_L_target + grad_L_XK_mini + tl.dot(grad_L_Z1.to(mp_dtype), tl.trans(W1_init).to(mp_dtype))

    # grads to params
    grad_L_W2_init = grad_L_W2_last + tl.dot(tl.trans(X2.to(mp_dtype)), grad_L_Z2.to(mp_dtype)) + grad_L_W2_init_extra
    grad_L_b2_init = grad_L_b2_last + tl.sum(grad_L_Z2, axis=0)[None, :]
    grad_L_W1_init = grad_L_W1_last + tl.dot(tl.trans(XK.to(mp_dtype)), grad_L_Z1.to(mp_dtype))
    grad_L_b1_init = grad_L_b1_last + tl.sum(grad_L_Z1, axis=0)[None, :]

    grad_L_norm_w = (grad_L_ln_weight_ln + grad_L_ln_weight_fused)[None, :]
    grad_L_norm_b = (grad_L_ln_bias_ln + grad_L_ln_bias_fused)[None, :]

    return (
        grad_L_norm_w,
        grad_L_norm_b,
        grad_L_W1_init,
        grad_L_b1_init,
        grad_L_W2_init,
        grad_L_b2_init,
        grad_L_XQ,
        grad_L_XV,
        grad_L_XK,
        grad_L_eta,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
    ],
    key=["checkpoint_group_size", "FF"],
)
@triton.jit
def ttt_mlp_scan_backward(
    XQ_batch_ptr,
    XV_batch_ptr,
    XK_batch_ptr,
    eta_batch_ptr,
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W1_checkpoints_ptr,
    b1_checkpoints_ptr,
    W2_checkpoints_ptr,
    b2_checkpoints_ptr,
    # Upstream
    grad_L_W1_last_ptr,
    grad_L_b1_last_ptr,
    grad_L_W2_last_ptr,
    grad_L_b2_last_ptr,
    grad_L_XQW_ptr,
    # Group buffers
    W1_init_group_ptr,
    W2_init_group_ptr,
    x_hat_ln_group_ptr,
    std_ln_group_ptr,
    Z1_group_ptr,
    Z1_bar_group_ptr,
    X2_group_ptr,
    X2_bar_group_ptr,
    grad_l_wrt_Z1_group_ptr,
    grad_l_wrt_Z2_group_ptr,
    x_hat_fused_group_ptr,
    grad_x_hat_fused_group_ptr,
    grad_output_fused_group_ptr,
    std_fused_group_ptr,
    # Outputs
    grad_L_ttt_norm_weight_ptr,
    grad_L_ttt_norm_bias_ptr,
    grad_L_W1_init_ptr,
    grad_L_b1_init_ptr,
    grad_L_W2_init_ptr,
    grad_L_b2_init_ptr,
    grad_L_XQ_ptr,
    grad_L_XV_ptr,
    grad_L_XK_ptr,
    grad_L_eta_ptr,
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
    batch = tl.program_id(0)
    head = tl.program_id(1)
    mp_dtype = XQ_batch_ptr.type.element_ty

    # checkpoint strides
    K_F_FF_stride = K * F_FF_stride
    K_FF_stride = K * FF_stride
    K_FF_F_stride = K * FF_F_stride
    K_F_stride = K * F_stride
    CS_stride = CS

    # offsets for grads (final)
    W1_off = batch * NH * F_FF_stride + head * F_FF_stride + tl.arange(0, F)[:, None] * FF + tl.arange(0, FF)[None, :]
    b1_off = batch * NH * FF_stride + head * FF_stride + tl.arange(0, FF)[None, :]
    W2_off = batch * NH * FF_F_stride + head * FF_F_stride + tl.arange(0, FF)[:, None] * F + tl.arange(0, F)[None, :]
    b2_off = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    norm_off = head * F_stride + tl.arange(0, F)
    norm_store_off = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_off).to(tl.float32)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_off).to(tl.float32)[None, :]

    # upstream
    grad_L_W1_last = tl.load(grad_L_W1_last_ptr + W1_off).to(tl.float32)
    grad_L_b1_last = tl.load(grad_L_b1_last_ptr + b1_off).to(tl.float32)
    grad_L_W2_last = tl.load(grad_L_W2_last_ptr + W2_off).to(tl.float32)
    grad_L_b2_last = tl.load(grad_L_b2_last_ptr + b2_off).to(tl.float32)

    grad_L_norm_w = tl.zeros((1, F), dtype=tl.float32)
    grad_L_norm_b = tl.zeros((1, F), dtype=tl.float32)

    for checkpoint_idx in range(K - 1, -1, -1):
        W1_ckpt_off = (
            batch * NH * K_F_FF_stride
            + head * K_F_FF_stride
            + checkpoint_idx * F_FF_stride
            + tl.arange(0, F)[:, None] * FF
            + tl.arange(0, FF)[None, :]
        )
        b1_ckpt_off = (
            batch * NH * K_FF_stride
            + head * K_FF_stride
            + checkpoint_idx * FF_stride
            + tl.arange(0, FF)[None, :]
        )
        W2_ckpt_off = (
            batch * NH * K_FF_F_stride
            + head * K_FF_F_stride
            + checkpoint_idx * FF_F_stride
            + tl.arange(0, FF)[:, None] * F
            + tl.arange(0, F)[None, :]
        )
        b2_ckpt_off = (
            batch * NH * K_F_stride
            + head * K_F_stride
            + checkpoint_idx * F_stride
            + tl.arange(0, F)[None, :]
        )

        W1_init = tl.load(W1_checkpoints_ptr + W1_ckpt_off).to(tl.float32)
        b1_init = tl.load(b1_checkpoints_ptr + b1_ckpt_off).to(tl.float32)
        W2_init = tl.load(W2_checkpoints_ptr + W2_ckpt_off).to(tl.float32)
        b2_init = tl.load(b2_checkpoints_ptr + b2_ckpt_off).to(tl.float32)

        # forward within group (save intermediates)
        for j in range(0, checkpoint_group_size):
            i = checkpoint_idx * checkpoint_group_size + j
            if i < NC:
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
                last_eta = tl.load(eta_batch_ptr + last_CS_off).to(tl.float32)

                (
                    _XQW,
                    W1_curr,
                    b1_curr,
                    W2_curr,
                    b2_curr,
                    Z1,
                    Z1_bar,
                    X2,
                    X2_bar,
                    x_hat_ln,
                    std_ln,
                    grad_Z1,
                    grad_Z2,
                    x_hat_fused,
                    grad_xhat_fused,
                    grad_out_fused,
                    std_fused,
                ) = ttt_mlp_mini_batch_forward(
                    W1_init,
                    b1_init,
                    W2_init,
                    b2_init,
                    ln_weight,
                    ln_bias,
                    XQ,
                    XK,
                    XV,
                    last_eta,
                    CS,
                    F,
                    FF,
                    mp_dtype,
                )

                # group offsets
                G_W1_off = (
                    batch * NH * checkpoint_group_size * F_FF_stride
                    + head * checkpoint_group_size * F_FF_stride
                    + j * F_FF_stride
                    + tl.arange(0, F)[:, None] * FF
                    + tl.arange(0, FF)[None, :]
                )
                G_W2_off = (
                    batch * NH * checkpoint_group_size * FF_F_stride
                    + head * checkpoint_group_size * FF_F_stride
                    + j * FF_F_stride
                    + tl.arange(0, FF)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                G_CS_F_off = (
                    batch * NH * checkpoint_group_size * CS_F_stride
                    + head * checkpoint_group_size * CS_F_stride
                    + j * CS_F_stride
                    + tl.arange(0, CS)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                G_CS_FF_off = (
                    batch * NH * checkpoint_group_size * CS_FF_stride
                    + head * checkpoint_group_size * CS_FF_stride
                    + j * CS_FF_stride
                    + tl.arange(0, CS)[:, None] * FF
                    + tl.arange(0, FF)[None, :]
                )
                G_CS_off = (
                    batch * NH * checkpoint_group_size * CS_stride
                    + head * checkpoint_group_size * CS_stride
                    + j * CS_stride
                    + tl.arange(0, CS)[:, None]
                )

                tl.store(W1_init_group_ptr + G_W1_off, W1_init)
                tl.store(W2_init_group_ptr + G_W2_off, W2_init)

                tl.store(Z1_group_ptr + G_CS_FF_off, Z1)
                tl.store(Z1_bar_group_ptr + G_CS_FF_off, Z1_bar)
                tl.store(X2_group_ptr + G_CS_FF_off, X2)
                tl.store(X2_bar_group_ptr + G_CS_FF_off, X2_bar)

                tl.store(x_hat_ln_group_ptr + G_CS_F_off, x_hat_ln)
                tl.store(std_ln_group_ptr + G_CS_off, std_ln)

                tl.store(grad_l_wrt_Z1_group_ptr + G_CS_FF_off, grad_Z1)
                tl.store(grad_l_wrt_Z2_group_ptr + G_CS_F_off, grad_Z2)

                tl.store(x_hat_fused_group_ptr + G_CS_F_off, x_hat_fused)
                tl.store(grad_x_hat_fused_group_ptr + G_CS_F_off, grad_xhat_fused)
                tl.store(grad_output_fused_group_ptr + G_CS_F_off, grad_out_fused)
                tl.store(std_fused_group_ptr + G_CS_off, std_fused)

                W1_init, b1_init, W2_init, b2_init = W1_curr, b1_curr, W2_curr, b2_curr

        W1_last = W1_init
        W2_last = W2_init

        # backward within group
        for j in range(checkpoint_group_size - 1, -1, -1):
            i = checkpoint_idx * checkpoint_group_size + j
            if i < NC:
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

                XQ = tl.load(XQ_batch_ptr + CS_F_off)
                XK = tl.load(XK_batch_ptr + CS_F_off).to(tl.float32)
                grad_L_XQW = tl.load(grad_L_XQW_ptr + CS_F_off).to(tl.float32)
                last_eta = tl.load(eta_batch_ptr + last_CS_off).to(tl.float32)

                # group offsets
                G_W1_off = (
                    batch * NH * checkpoint_group_size * F_FF_stride
                    + head * checkpoint_group_size * F_FF_stride
                    + j * F_FF_stride
                    + tl.arange(0, F)[:, None] * FF
                    + tl.arange(0, FF)[None, :]
                )
                G_W2_off = (
                    batch * NH * checkpoint_group_size * FF_F_stride
                    + head * checkpoint_group_size * FF_F_stride
                    + j * FF_F_stride
                    + tl.arange(0, FF)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                G_CS_F_off = (
                    batch * NH * checkpoint_group_size * CS_F_stride
                    + head * checkpoint_group_size * CS_F_stride
                    + j * CS_F_stride
                    + tl.arange(0, CS)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                G_CS_FF_off = (
                    batch * NH * checkpoint_group_size * CS_FF_stride
                    + head * checkpoint_group_size * CS_FF_stride
                    + j * CS_FF_stride
                    + tl.arange(0, CS)[:, None] * FF
                    + tl.arange(0, FF)[None, :]
                )
                G_CS_off = (
                    batch * NH * checkpoint_group_size * CS_stride
                    + head * checkpoint_group_size * CS_stride
                    + j * CS_stride
                    + tl.arange(0, CS)[:, None]
                )

                W1_curr = tl.load(W1_init_group_ptr + G_W1_off).to(tl.float32)
                W2_curr = tl.load(W2_init_group_ptr + G_W2_off).to(tl.float32)

                Z1 = tl.load(Z1_group_ptr + G_CS_FF_off).to(tl.float32)
                Z1_bar = tl.load(Z1_bar_group_ptr + G_CS_FF_off).to(tl.float32)
                X2 = tl.load(X2_group_ptr + G_CS_FF_off).to(tl.float32)
                X2_bar = tl.load(X2_bar_group_ptr + G_CS_FF_off).to(tl.float32)

                x_hat_ln = tl.load(x_hat_ln_group_ptr + G_CS_F_off).to(tl.float32)
                std_ln = tl.load(std_ln_group_ptr + G_CS_off).to(tl.float32)

                grad_Z1 = tl.load(grad_l_wrt_Z1_group_ptr + G_CS_FF_off).to(tl.float32)
                grad_Z2 = tl.load(grad_l_wrt_Z2_group_ptr + G_CS_F_off).to(tl.float32)

                grad_out_fused = tl.load(grad_output_fused_group_ptr + G_CS_F_off).to(tl.float32)
                std_fused = tl.load(std_fused_group_ptr + G_CS_off).to(tl.float32)

                x_hat_fused = tl.load(x_hat_fused_group_ptr + G_CS_F_off).to(mp_dtype)
                grad_xhat_fused = tl.load(grad_x_hat_fused_group_ptr + G_CS_F_off).to(mp_dtype)

                (
                    dnorm_w_mb,
                    dnorm_b_mb,
                    dW1,
                    db1,
                    dW2,
                    db2,
                    dXQ,
                    dXV,
                    dXK,
                    dEta,
                ) = ttt_mlp_mini_batch_backward(
                    XQ,
                    XK,
                    W1_curr,
                    W1_last,
                    W2_curr,
                    W2_last,
                    Z1,
                    Z1_bar,
                    X2,
                    X2_bar,
                    ln_weight,
                    std_fused,
                    x_hat_fused,
                    grad_out_fused,
                    grad_xhat_fused,
                    grad_Z1,
                    grad_Z2,
                    last_eta,
                    std_ln,
                    x_hat_ln,
                    grad_L_W1_last,
                    grad_L_b1_last,
                    grad_L_W2_last,
                    grad_L_b2_last,
                    grad_L_XQW,
                    CS,
                    F,
                    FF,
                    mp_dtype,
                )

                tl.store(grad_L_XQ_ptr + CS_F_off, dXQ)
                tl.store(grad_L_XV_ptr + CS_F_off, dXV)
                tl.store(grad_L_XK_ptr + CS_F_off, dXK)
                tl.store(grad_L_eta_ptr + CS_CS_off, dEta)

                grad_L_W1_last = dW1
                grad_L_b1_last = db1
                grad_L_W2_last = dW2
                grad_L_b2_last = db2

                grad_L_norm_w += dnorm_w_mb
                grad_L_norm_b += dnorm_b_mb

                W1_last = W1_curr
                W2_last = W2_curr

    tl.store(grad_L_ttt_norm_weight_ptr + norm_store_off, grad_L_norm_w)
    tl.store(grad_L_ttt_norm_bias_ptr + norm_store_off, grad_L_norm_b)

    tl.store(grad_L_W1_init_ptr + W1_off, grad_L_W1_last)
    tl.store(grad_L_b1_init_ptr + b1_off, grad_L_b1_last)
    tl.store(grad_L_W2_init_ptr + W2_off, grad_L_W2_last)
    tl.store(grad_L_b2_init_ptr + b2_off, grad_L_b2_last)
