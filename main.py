import torch
import torch.nn.functional as F
from types import SimpleNamespace

from data_factory import data_provider
from forecast import ModelConfig
from forecast import TTTForecaster

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- args like your other repo expects ----
    args = SimpleNamespace(
        # dataset
        task_name="short_term_forecast",
        data="ETTh1",
        root_path="ETT",      # folder containing ETTh1.csv
        data_path="ETTh1.csv",
        features="S",                  # "S", "M", or "MS"
        target="OT",
        freq="h",
        embed="timeF",                 # controls timeenc in data_provider
        seasonal_patterns=None,

        # lengths
        seq_len=384,
        label_len=96,
        pred_len=96,

        # loader
        batch_size=32,
        num_workers=0,

        # your Dataset_ETT_hour checks this
        augmentation_ratio=0,
    )

    train_set, train_loader = data_provider(args, "train")
    val_set, val_loader = data_provider(args, "val")

    # Peek one batch to infer dims safely
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
    d_in = batch_x.shape[-1]  # number of input features/channels

    # Same convention as your other code:
    # MS => multivariate input, single target output (last dim)
    f_dim = -1 if args.features == "MS" else 0
    d_out = 1 if args.features == "MS" else batch_y.shape[-1]

    # ---- config for TTT ----
    cfg = ModelConfig(
        seq_len=args.seq_len,
        model_dim=128,
        num_heads=4,
        num_layers=1,
        ssm_layer="ttt_mlp",        # or "ttt_mlp"
        mini_batch_size=64,            # must divide seq_len
        ttt_base_lr=0.05,
        scan_checkpoint_group_size=16,
        latent_height=1,
        latent_width=1,
        compressed_num_frames=None,    # becomes seq_len in __post_init__
    )

    model = TTTForecaster(cfg, d_in=d_in, d_out=d_out, pred_len=args.pred_len).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def run_epoch(loader, train: bool):
        model.train(train)
        total = 0.0
        n = 0
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)

            # target slice EXACTLY like your other code
            y_true = batch_y[:, -args.pred_len:, f_dim:]  # (B, pred_len, d_out)

            if train:
                opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_pred = model(batch_x)                  # (B, pred_len, d_out)
                loss = F.mse_loss(y_pred, y_true)

            if train:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

            total += loss.item() * batch_x.size(0)
            n += batch_x.size(0)

        return total / max(n, 1)

    for epoch in range(20):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        print(f"epoch {epoch:02d} | train {train_loss:.6f} | val {val_loss:.6f}")

    torch.save(model.state_dict(), "ttt_forecaster.pt")
    print("saved ttt_forecaster.pt")

if __name__ == "__main__":
    main()
