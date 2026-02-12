import math
from functools import partial

import torch
from torch.distributed._tensor import Shard
from torch.distributed._tensor.experimental import local_map

from kernels.mlp_backward import ttt_mlp_scan_backward
from kernels.mlp_forward import ttt_mlp_scan_forward


class TritonMLP(torch.autograd.Function):
    sharded_mode = False

    @staticmethod
    def forward(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ) -> torch.Tensor:
        if TritonMLP.sharded_mode:
            return TritonMLP.forward_sharded(
                ctx,
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                W2_init,
                b2_init,
                XQ_batch,
                XV_batch,
                XK_batch,
                eta_batch,
                checkpoint_group_size,
            )
        else:
            return TritonMLP.forward_unsharded(
                ctx,
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                W2_init,
                b2_init,
                XQ_batch,
                XV_batch,
                XK_batch,
                eta_batch,
                checkpoint_group_size,
            )

    @staticmethod
    def backward(ctx, grad_L_XQW_batch):
        if TritonMLP.sharded_mode:
            return TritonMLP.backward_sharded(ctx, grad_L_XQW_batch)
        else:
            return TritonMLP.backward_unsharded(ctx, grad_L_XQW_batch)

    @staticmethod
    def _forward_core(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        B, NH, NC, CS, F = XQ_batch.shape
        FF = W1_init.shape[-1]  # 4*F for MLP

        # sanity
        assert W1_init.shape[-2] == F
        assert b1_init.shape[-1] == FF
        assert W2_init.shape[-2] == FF and W2_init.shape[-1] == F
        assert b2_init.shape[-1] == F

        K = math.ceil(NC / checkpoint_group_size)

        device = XQ_batch.device
        mp_dtype = XQ_batch.dtype

        # Outputs
        W1_last = torch.empty(B, NH, F, FF, device=device, dtype=torch.float32)
        b1_last = torch.empty(B, NH, 1, FF, device=device, dtype=torch.float32)
        W2_last = torch.empty(B, NH, FF, F, device=device, dtype=torch.float32)
        b2_last = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)
        XQW_batch = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)

        # Checkpoints
        W1_checkpoints = torch.empty(B, NH, K, F, FF, device=device, dtype=torch.float32)
        b1_checkpoints = torch.empty(B, NH, K, 1, FF, device=device, dtype=torch.float32)
        W2_checkpoints = torch.empty(B, NH, K, FF, F, device=device, dtype=torch.float32)
        b2_checkpoints = torch.empty(B, NH, K, 1, F, device=device, dtype=torch.float32)

        # Strides
        CS_F_stride = CS * F
        CS_FF_stride = CS * FF
        F_FF_stride = F * FF
        FF_F_stride = FF * F
        CS_CS_stride = CS * CS
        F_stride = F
        FF_stride = FF

        grid = (B, NH)

        ttt_mlp_scan_forward[grid](
            # Inputs
            ttt_norm_weight.contiguous(),
            ttt_norm_bias.contiguous(),
            W1_init.to(torch.float32).contiguous(),
            b1_init.to(torch.float32).contiguous(),
            W2_init.to(torch.float32).contiguous(),
            b2_init.to(torch.float32).contiguous(),
            XQ_batch.contiguous(),
            XV_batch.contiguous(),
            XK_batch.contiguous(),
            eta_batch.contiguous(),
            # Outputs
            W1_last.contiguous(),
            b1_last.contiguous(),
            W2_last.contiguous(),
            b2_last.contiguous(),
            XQW_batch.contiguous(),
            # Checkpoints
            W1_checkpoints.contiguous(),
            b1_checkpoints.contiguous(),
            W2_checkpoints.contiguous(),
            b2_checkpoints.contiguous(),
            # Strides
            CS_F_stride,
            CS_FF_stride,
            F_FF_stride,
            FF_F_stride,
            CS_CS_stride,
            F_stride,
            FF_stride,
            # Constexpr
            NH,
            NC,
            CS,
            F,
            FF,
            K,
            checkpoint_group_size,
        )

        checkpoint_shapes = torch.tensor([K, checkpoint_group_size])

        ctx.save_for_backward(
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
            checkpoint_shapes,
        )

        return XQW_batch.to(mp_dtype)

    @staticmethod
    def _backward_core(ctx, grad_L_XQW_batch):
        (
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
            checkpoint_shapes,
        ) = ctx.saved_tensors

        B, NH, NC, CS, F = XQ_batch.shape
        FF = W1_checkpoints.shape[-1]
        K, checkpoint_group_size = checkpoint_shapes[0].item(), checkpoint_shapes[1].item()

        device = XQ_batch.device
        mp_dtype = XQ_batch.dtype
        intermediate_dtype = torch.float32

        # Group buffers
        W1_init_group = torch.empty(B, NH, checkpoint_group_size, F, FF, device=device, dtype=torch.float32)
        W2_init_group = torch.empty(B, NH, checkpoint_group_size, FF, F, device=device, dtype=torch.float32)

        grad_L_W1_last = torch.zeros(B, NH, F, FF, device=device, dtype=torch.float32)
        grad_L_b1_last = torch.zeros(B, NH, 1, FF, device=device, dtype=torch.float32)
        grad_L_W2_last = torch.zeros(B, NH, FF, F, device=device, dtype=torch.float32)
        grad_L_b2_last = torch.zeros(B, NH, 1, F, device=device, dtype=torch.float32)

        x_hat_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype)
        std_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=intermediate_dtype)

        Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, FF, device=device, dtype=intermediate_dtype)
        Z1_bar_group = torch.empty(B, NH, checkpoint_group_size, CS, FF, device=device, dtype=intermediate_dtype)
        X2_group = torch.empty(B, NH, checkpoint_group_size, CS, FF, device=device, dtype=intermediate_dtype)
        X2_bar_group = torch.empty(B, NH, checkpoint_group_size, CS, FF, device=device, dtype=intermediate_dtype)

        grad_l_wrt_Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, FF, device=device, dtype=intermediate_dtype)
        grad_l_wrt_Z2_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype)

        x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype)
        grad_x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype)
        grad_output_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype)
        std_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=intermediate_dtype)

        # Outputs
        grad_L_ttt_norm_weight = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)
        grad_L_ttt_norm_bias = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)

        grad_L_W1_init = torch.empty(B, NH, F, FF, device=device, dtype=torch.float32)
        grad_L_b1_init = torch.empty(B, NH, 1, FF, device=device, dtype=torch.float32)
        grad_L_W2_init = torch.empty(B, NH, FF, F, device=device, dtype=torch.float32)
        grad_L_b2_init = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)

        grad_L_XQ = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)
        grad_L_XV = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)
        grad_L_XK = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)
        grad_L_eta = torch.empty(B, NH, NC, CS, CS, device=device, dtype=torch.float32)

        # Strides
        CS_F_stride = CS * F
        CS_FF_stride = CS * FF
        F_FF_stride = F * FF
        FF_F_stride = FF * F
        CS_CS_stride = CS * CS
        F_stride = F
        FF_stride = FF

        grid = (B, NH)

        ttt_mlp_scan_backward[grid](
            XQ_batch.contiguous(),
            XV_batch.contiguous(),
            XK_batch.contiguous(),
            eta_batch.contiguous(),
            ttt_norm_weight.contiguous(),
            ttt_norm_bias.contiguous(),
            W1_checkpoints.contiguous(),
            b1_checkpoints.contiguous(),
            W2_checkpoints.contiguous(),
            b2_checkpoints.contiguous(),
            # Upstream
            grad_L_W1_last.contiguous(),
            grad_L_b1_last.contiguous(),
            grad_L_W2_last.contiguous(),
            grad_L_b2_last.contiguous(),
            grad_L_XQW_batch.contiguous(),
            # Group buffers
            W1_init_group.contiguous(),
            W2_init_group.contiguous(),
            x_hat_ln_group.contiguous(),
            std_ln_group.contiguous(),
            Z1_group.contiguous(),
            Z1_bar_group.contiguous(),
            X2_group.contiguous(),
            X2_bar_group.contiguous(),
            grad_l_wrt_Z1_group.contiguous(),
            grad_l_wrt_Z2_group.contiguous(),
            x_hat_fused_group.contiguous(),
            grad_x_hat_fused_group.contiguous(),
            grad_output_fused_group.contiguous(),
            std_fused_group.contiguous(),
            # Outputs
            grad_L_ttt_norm_weight.contiguous(),
            grad_L_ttt_norm_bias.contiguous(),
            grad_L_W1_init.contiguous(),
            grad_L_b1_init.contiguous(),
            grad_L_W2_init.contiguous(),
            grad_L_b2_init.contiguous(),
            grad_L_XQ.contiguous(),
            grad_L_XV.contiguous(),
            grad_L_XK.contiguous(),
            grad_L_eta.contiguous(),
            # Strides
            CS_F_stride,
            CS_FF_stride,
            F_FF_stride,
            FF_F_stride,
            CS_CS_stride,
            F_stride,
            FF_stride,
            # Constexpr
            NH,
            NC,
            CS,
            F,
            FF,
            K,
            checkpoint_group_size,
        )

        grad_L_ttt_norm_weight = grad_L_ttt_norm_weight.sum(dim=0).squeeze(1)
        grad_L_ttt_norm_bias = grad_L_ttt_norm_bias.sum(dim=0).squeeze(1)

        return (
            grad_L_ttt_norm_weight.to(mp_dtype),
            grad_L_ttt_norm_bias.to(mp_dtype),
            grad_L_W1_init.to(mp_dtype),
            grad_L_b1_init.to(mp_dtype),
            grad_L_W2_init.to(mp_dtype),
            grad_L_b2_init.to(mp_dtype),
            grad_L_XQ.to(mp_dtype),
            grad_L_XV.to(mp_dtype),
            grad_L_XK.to(mp_dtype),
            grad_L_eta.to(mp_dtype),
            None,
            None,
        )

    @staticmethod
    @partial(
        local_map,
        in_placements=(
            None,
            [Shard(0)],
            [Shard(0)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            None,
        ),
        out_placements=([Shard(1)],),
    )
    def forward_sharded(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        return TritonMLP._forward_core(
            ctx,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            W2_init,
            b2_init,
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            checkpoint_group_size,
        )

    @staticmethod
    @partial(local_map, in_placements=None, out_placements=None)
    def forward_unsharded(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        return TritonMLP._forward_core(
            ctx,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            W2_init,
            b2_init,
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            checkpoint_group_size,
        )

    @staticmethod
    @partial(
        local_map,
        in_placements=(None, [Shard(1)]),
        out_placements=(
            [Shard(0)],
            [Shard(0)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            [Shard(1)],
            None,
            None,
        ),
    )
    def backward_sharded(ctx, grad_L_XQW_batch):
        return TritonMLP._backward_core(ctx, grad_L_XQW_batch)

    @staticmethod
    @partial(local_map, in_placements=None, out_placements=None)
    def backward_unsharded(ctx, grad_L_XQW_batch):
        return TritonMLP._backward_core(ctx, grad_L_XQW_batch)