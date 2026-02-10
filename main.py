import torch

# Import from the repo
from linear_triton import TritonLinear  # path may differ in your repo

device = "cuda"
B, T, D_in = 4, 256, 8
NH = 1
F = 64
CS = 32
NC = (T + CS - 1) // CS
T_pad = NC * CS

# --- fake time series ---
x = torch.randn(B, T, D_in, device=device, dtype=torch.bfloat16)

# pad to multiple of CS
if T_pad != T:
    pad = torch.zeros(B, T_pad - T, D_in, device=device, dtype=x.dtype)
    x = torch.cat([x, pad], dim=1)

# embed to F
embed = torch.nn.Linear(D_in, F, bias=False).to(device, dtype=torch.bfloat16)
h = embed(x)  # (B, T_pad, F)

# chunk + add head dim => (B, NH, NC, CS, F)
h = h.view(B, NC, CS, F).unsqueeze(1)

# choose Q,K,V (simple: all the same)
XQ_batch = h
XK_batch = h
XV_batch = h

# norm params per head
ttt_norm_weight = torch.ones(NH, F, device=device, dtype=torch.bfloat16)
ttt_norm_bias   = torch.zeros(NH, F, device=device, dtype=torch.bfloat16)

# initial fast weights per (B, head)
W1_init = torch.zeros(B, NH, F, F, device=device, dtype=torch.bfloat16)
b1_init = torch.zeros(B, NH, 1, F, device=device, dtype=torch.bfloat16)

# eta_batch: (B, NH, NC, CS, CS)
# kernel uses only last row eta[..., CS-1, :] as per-token step sizes
eta_batch = torch.zeros(B, NH, NC, CS, CS, device=device, dtype=torch.bfloat16)
eta_batch[..., CS-1, :] = 1e-3  # constant step size for all positions in chunk

checkpoint_group_size = 4  # trade-off memory vs recompute in backward

# run the adaptive layer
XQW_batch = TritonLinear.apply(
    ttt_norm_weight, ttt_norm_bias,
    W1_init, b1_init,
    XQ_batch, XV_batch, XK_batch,
    eta_batch,
    checkpoint_group_size,
)

# back to (B, T_pad, F)
out = XQW_batch.squeeze(1).reshape(B, T_pad, F)

# toy objective (predict next-step embedding)
loss = (out[:, :-1] - out[:, 1:]).pow(2).mean()
loss.backward()
print("ok", loss.item())
