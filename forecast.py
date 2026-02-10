from dataclasses import dataclass


def to_local(x): 
    return x

def place_into(x, like): 
    return x

def full_tensor(x): 
    return x

def shard_tensor(x, mesh, dim): 
    return x

@dataclass
class SequenceMetadata:
    seq_text_length: int = 0
    is_multiscene: bool = False
    # the rest are unused when is_multiscene=False
    init_offset: int | None = None
    base_offset: int | None = None
    num_chunks: int = 1
    text_length: int = 0


import json
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    seq_len: int = 384
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1

    ssm_layer: str = "ttt_linear"
    layer_norm_eps: float = 1e-6

    # TTT-Specific Configs
    mini_batch_size: int = 64
    ttt_base_lr: float = 0.1

    rope_theta: float = 10000.0
    scan_checkpoint_group_size: int = 16

    # ROPE Config
    latent_height: int = 1
    latent_width: int = 1
    theta: float = 10000
    compressed_num_frames: int | None = None

    def __post_init__(self):
        if self.compressed_num_frames is None:
            self.compressed_num_frames = self.seq_len
        assert self.model_dim % self.num_heads == 0
        assert self.seq_len % self.mini_batch_size == 0
        assert (self.model_dim // self.num_heads) % 2 == 0  # RoPE needs even head_dim


import torch
import torch.nn as nn
import torch.nn.functional as F

# use the real type from your repo if available
from ttt_layer import TTTWrapper  # wherever your class lives

class TTTForecaster(nn.Module):
    def __init__(self, config, d_in, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.in_proj = nn.Linear(d_in, config.model_dim)
        self.ttt = TTTWrapper(config)
        self.ttt.init_freqs()
        self.head = nn.Linear(config.model_dim, pred_len)
        self.meta = SequenceMetadata(seq_text_length=0, is_multiscene=False)
        

    def forward(self, seq_x):
        # seq_x: (B, L, d_in)
        h = self.in_proj(seq_x)
        h = self.ttt(h, self.meta)      # (B, L, model_dim)
        yhat = self.head(h[:, -1]) # (B, pred_len)
        return yhat
