# train_ttt_forecaster.py
# Assumes you already have:
# - Dataset_ETT_hour (your dataloader class)
# - TTTForecaster, ModelConfig (from your forecast.py)
# - ETTh1.csv downloaded

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from forecast import ModelConfig, TTTForecaster
from data_loader import Dataset_ETT_hour  # <- change to your file name

# ---------- minimal args stub for your dataset ----------
class Args:
    augmentation_ratio = 0  # dataset checks this

# ---------- training loop ----------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset paths
    root_path = "ETT"         # <- folder containing ETTh1.csv
    data_path = "ETTh1.csv"

    # Build dataset + loader
    args = Args()
    train_set = Dataset_ETT_hour(
        args=args,
        root_path=root_path,
        flag="train",
        features="S",          # 'S' => single target, easiest sanity
        data_path=data_path,
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    )
    batch_size = 32
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    # Config (must match dataset seq_len)
    seq_len = train_set.seq_len
    pred_len = train_set.pred_len
    cfg = ModelConfig(
        seq_len=seq_len,
        model_dim=128,
        num_heads=4,
        num_layers=1,
        ssm_layer="ttt_linear",
        mini_batch_size=64,            # must divide seq_len (384 ok)
        ttt_base_lr=0.05,              # slightly safer than 0.1
        scan_checkpoint_group_size=16,
        latent_height=1,
        latent_width=1,
        compressed_num_frames=None,    # will become seq_len in __post_init__
    )

    # Model
    d_in = 1  # because features='S'
    model = TTTForecaster(cfg, d_in=d_in, pred_len=pred_len).to(device)

    # Mixed precision (optional)
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    model.train()
    global_step = 0
    for epoch in range(3):
        for batch in train_loader:
            seq_x, seq_y, seq_x_mark, seq_y_mark = batch

            # seq_x: (B, seq_len, 1)
            # seq_y: (B, label_len + pred_len, 1)
            seq_x = seq_x.to(device, non_blocking=True).float()
            seq_y = seq_y.to(device, non_blocking=True).float()

            # Predict the next pred_len points (target is last pred_len of seq_y)
            y_true = seq_y[:, -pred_len:, 0]  # (B, pred_len)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_pred = model(seq_x)         # (B, pred_len)
                loss = F.mse_loss(y_pred, y_true)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if global_step % 50 == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")

            global_step += 1

        # quick save
        torch.save({"model": model.state_dict(), "cfg": cfg}, f"ttt_forecaster_epoch{epoch}.pt")

if __name__ == "__main__":
    train()
