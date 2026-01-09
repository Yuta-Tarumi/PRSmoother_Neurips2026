import math
import torch.nn as nn
import torch
from typing import List, Literal, Sequence
#import transformer_engine.pytorch as te
#from transformer_engine.common.recipe import Format, DelayedScaling
#fp8_recipe = DelayedScaling(amax_history_len=16, fp8_format=Format.HYBRID)

from torch import Tensor
import torch.nn.functional as F

__all__ = ["cpad", "Conv1DRecognition"]

class ScaledTanhHead(nn.Module):
    def __init__(self, in_ch, out_ch, init_bound=0.05, learnable=True):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # y in [-s, s] initially
        v = math.log(init_bound)
        self.log_s = nn.Parameter(torch.tensor(v)) if learnable else nn.Parameter(torch.tensor(v), requires_grad=False)

    def forward(self, x):
        y = self.proj(x)
        s = self.log_s.exp()
        # smooth, gradient-friendly bounding
        return s * torch.tanh(y / s)

# --- Periodic Fourier positional encoding (length L, dim C) -------------------
class FourierPositionalEncoding1D(nn.Module):
    """Sin/Cos features with frequencies tied to the circular domain length L."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)  -> returns (1, L, C)
        L = x.size(1)
        device = x.device
        half = self.dim // 2
        pos = torch.arange(L, device=device).float()[:, None]              # (L,1)
        k = torch.arange(1, half + 1, device=device).float()[None, :]      # (1,half)
        ang = 2.0 * math.pi * pos * k / L                                  # (L,half)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)           # (L,2*half)
        if self.dim % 2 == 1:
            pe = F.pad(pe, (0, 1))                                         # (L,dim)
        return pe.unsqueeze(0)                                             # (1,L,dim)

# --- Self-attention block in (B,C,L), pre-norm, residual ----------------------
class AttnBlock1D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0, use_posenc: bool = True, ffn_expand: int = 2):
        """
        dim must be divisible by num_heads.
        """
        super().__init__()
        assert dim % num_heads == 0, "embed dim must be divisible by num_heads"
        self.use_posenc = use_posenc
        self.posenc = FourierPositionalEncoding1D(dim) if use_posenc else None

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           dropout=dropout, batch_first=True)
        self.drop  = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, ffn_expand * dim),
            nn.GELU(),
            nn.Linear(ffn_expand * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L) -> (B,L,C) for attention
        B, C, L = x.shape
        h = x.transpose(1, 2)                  # (B,L,C)
        y = self.norm1(h)
        if self.use_posenc:
            y = y + self.posenc(y)             # broadcast (1,L,C)
        y, _ = self.attn(y, y, y, need_weights=False)
        h = h + self.drop(y)                   # residual 1

        y = self.norm2(h)
        y = self.ffn(y)
        h = h + self.drop(y)                   # residual 2

        return h.transpose(1, 2)               # (B,C,L)

# -----------------------------------------------------------------------------
# Helper for circular‑padded 1‑D convolution
# -----------------------------------------------------------------------------

def cpad(k: int = 3):
    """Return kwargs for a 1‑D convolution with *circular* padding.

    *k* must be odd so that *padding=k//2* preserves the sequence length.
    """
    return dict(kernel_size=k, padding=k // 2, padding_mode="circular", bias=False)


# -----------------------------------------------------------------------------
# Residual block with LayerNorm (implemented via GroupNorm) and ReLU
# -----------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Conv → LayerNorm → ReLU with an optional skip connection.

    * If *in_channels == out_channels* the skip path is identity.
    * Otherwise a 1×1 convolution aligns the channel count.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **cpad(k))
        # LayerNorm for Conv1d: GroupNorm with 1 group behaves equivalently
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.ReLU(inplace=True)

        if in_channels == out_channels:
            self.skip = None  # identity
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        if self.skip is not None:
            x = self.skip(x)
        return y + x

class _ResidualFC(nn.Module):
    """Pre-norm residual MLP block: y = x + Act(Linear(LN(x)))"""
    def __init__(self, dim: int = 256, activation: str = "gelu"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc   = nn.Linear(dim, dim)
        self.act  = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.fc(self.norm(x)))

class LinearRecognition(nn.Module):
    """
    Fully-connected recognition network with LayerNorm + residual blocks.

    Input  : (B, steps, L=4)
    Output : (B, 64,   L=4)

    Pipeline:
      - Flatten to (B, 4*steps)
      - Linear(4*steps -> 256), LayerNorm, GELU/ReLU
      - Seven residual FC blocks of (256 -> 256), each with LayerNorm + skip
      - Final Linear(256 -> 64*4) without normalization; reshape to (B, 64, 4)
    """

    def __init__(self, steps: int = 1, out_channels: int = 64,
                 activation: str = "gelu"):
        super().__init__()
        self.steps = int(steps)
        self.L = 4
        self.hidden = 256

        if out_channels != 64:
            raise ValueError(f"This module outputs 64 channels; got out_channels={out_channels}.")

        # (4*steps) -> 256
        self.in_proj = nn.Linear(self.steps * self.L, self.hidden)
        self.in_norm = nn.LayerNorm(self.hidden)
        self.in_act  = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        # Seven residual 256->256 blocks
        self.blocks = nn.Sequential(
            _ResidualFC(self.hidden, activation=activation),
            _ResidualFC(self.hidden, activation=activation),
            _ResidualFC(self.hidden, activation=activation),
            _ResidualFC(self.hidden, activation=activation),
            _ResidualFC(self.hidden, activation=activation),
            _ResidualFC(self.hidden, activation=activation),
            _ResidualFC(self.hidden, activation=activation),
        )

        # Final head: 256 -> (64*4); no LayerNorm here
        self.out_proj = nn.Linear(self.hidden, 64 * self.L)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, steps, L=4)  →  returns (B, 64, 4)
        """
        B, S, L = y.shape
        if L != self.L:
            raise ValueError(f"Expected last dim L={self.L}, got {L}.")
        if S != self.steps:
            raise ValueError(f"Expected steps={self.steps}, got {S}.")

        x = y.reshape(B, S * L)    # (B, 4*steps)
        x = self.in_proj(x)        # (B, 256)
        x = self.in_norm(x)
        x = self.in_act(x)

        x = self.blocks(x)         # (B, 256)
        x = self.out_proj(x)       # (B, 64*4)
        x = x.view(B, 64, self.L)  # (B, 64, 4)
        return x


# -----------------------------------------------------------------------------
# Eight‑layer periodic Conv1d encoder with residual connections
# -----------------------------------------------------------------------------

class Conv1DRecognition(nn.Module):
    """Periodic 1‑D convolution encoder for Lorenz‑96 observations.

    Channel progression: 50 → 128 → 128 → 128 → 128 → 32 → 8 → 4 → 1
    with LayerNorm + ReLU after each convolution and residual (skip)
    connections whenever the channel size stays constant.

    Expected input shape  : *(B, 50, L)* where *L = 40*.
    Output feature shape : *(B, 1,  L)*.
    """

    #CHANNELS: List[int] = [2, 128, 128, 128, 128, 128, 32, 8, 2]
    #CHANNELS: List[int] = [1, 128, 128, 128, 128, 128, 32, 8, 2]

    def __init__(self, kernel_size: int = 3, steps: int = 1, out_channels: int=2):
        super().__init__()
        self.CHANNELS: List[int] = [steps, 128, 128, 128, 128, 128, 128, 128, out_channels]
        layers = []
        for c_in, c_out in zip(self.CHANNELS[:-1], self.CHANNELS[1:]):
            # Use residual block for all but the final layer (c_out == 1)
            if c_out != 1:
                layers.append(_ResBlock(c_in, c_out, k=kernel_size))
            else:
                # Final linear projection without activation / residual
                layers.append(nn.Conv1d(c_in, c_out, **cpad(kernel_size)))
        self.net = nn.Sequential(*layers)
        init_bound = 10
        print(f"{init_bound=}")
        self.head = ScaledTanhHead(in_ch=out_channels, out_ch=out_channels, init_bound=init_bound)

    def forward(self, y):
        """Input *(B, 50, L)* → output *(B, 1, L)*."""
        return self.head(self.net(y))

class Conv1DRecognition_Attn(nn.Module):
    """
    Periodic 1-D convolution encoder for Lorenz-96 observations with optional
    global self-attention along the state dimension L.

    Expected input: (B, steps, L)  e.g., steps=50, L=40
    Output:         (B, out_channels, L)

    Attention is inserted after selected conv blocks while channel dim=128.
    """
    def __init__(
        self,
        kernel_size: int = 3,
        steps: int = 50,
        out_channels: int = 2,
        # Insert attention after these conv block indices (0-based among hidden blocks)
        attn_at: Sequence[int] = (2, 5),
        attn_heads: int = 8,
        attn_dropout: float = 0.0,
        use_posenc: bool = True,
    ):
        super().__init__()
        self.CHANNELS: List[int] = [steps, 128, 128, 128, 128, 128, 128, 128, out_channels]
        print(f"{self.CHANNELS=}")
        layers: List[nn.Module] = []
        hidden_block_idx = 0  # counts only the hidden conv blocks (not the final projection)

        for i, (c_in, c_out) in enumerate(zip(self.CHANNELS[:-1], self.CHANNELS[1:])):
            is_last = (i == len(self.CHANNELS) - 2)

            if is_last:
                # Final linear projection without activation/residual
                layers.append(nn.Conv1d(c_in, c_out, kernel_size=1))
            else:
                # Residual conv block
                layers.append(_ResBlock(c_in, c_out, k=kernel_size))

                # Inject attention after chosen hidden blocks when dim is 128
                if (c_out == 128) and (hidden_block_idx in set(attn_at)):
                    layers.append(AttnBlock1D(dim=128, num_heads=attn_heads,
                                              dropout=attn_dropout, use_posenc=use_posenc))
                hidden_block_idx += 1

        self.net = nn.Sequential(*layers)
        init_bound = 10
        print(f"{init_bound=}")
        self.head = ScaledTanhHead(in_ch=out_channels, out_ch=out_channels, init_bound=init_bound)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, steps, L) -> (B, out_channels, L)
        return self.head(self.net(y))#self.net(y)

class ConvEmbed(nn.Module):
    def __init__(self, obs_dim=40, n_tokens=128, d_model=512, k=5):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=50,
            out_channels=n_tokens,
            kernel_size=k,
            padding=k//2,
            padding_mode="circular"
        )
        self.linear = nn.Linear(obs_dim, d_model)
    def forward(self, x):
        x = self.proj(x) # x: (B, T, 40) -> (B, n_tokens, 40)
        x = self.linear(x) # x: (B, n_tokens, 40) -> (B, n_tokens, d_model)
        return x  # back to (B, n_tokens, d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))           # (1, T, d)

    def forward(self, x):                                     # x: (B, T, d)
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    """Maps (B, 50, 40) → (B, 2, 40) via flatten-then-Linear."""
    def __init__(self, d_model: int = 512, n_layers: int = 8, obs_dim: int = 40, out_channels: int = 2, emb_arch: str = "Conv"):
        super().__init__()
        self.obs_dim = obs_dim
        if emb_arch == "Linear":
            self.embed  = nn.Linear(self.obs_dim, d_model)                  # tokeniser
        elif emb_arch == "Conv":
            self.embed  = ConvEmbed()
            
        self.posenc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4,
            dim_feedforward=2*d_model, dropout=0.0,
            batch_first=True, norm_first=True)
        
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        # Flatten (B, 50, d) → (B, 50*d) then project to 80 = 2×40
        self.out_channels = out_channels
        self.head = nn.Linear(64 * d_model, self.out_channels*40)

    def forward(self, x):                                     # x: (B, 50, 40)
        z = self.posenc(self.embed(x))                        # (B, 50, d)
        z = self.encoder(z)                                   # 8-layer stack
        z = z.flatten(1)                                      # (B, 50*d_model)
        y = self.head(z).view(-1, self.out_channels, 40)                      # (B, 2, 40)
        '''
        # ── 1. Pad time dimension to multiple of 16 ─────────────────────────
        T = x.size(1)
        if T % 16:                             # e.g. 50 → pad_len = 14 → 64
            pad_len = 16 - (T % 16)
            pad = torch.zeros(
                x.size(0), pad_len, self.obs_dim,
                dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, pad], dim=1)     # (B, T+pad_len, 40)

        # ── 2. Normal forward ───────────────────────────────────────────────
        z = self.posenc(self.embed(x))         # (B, T_pad, d_model)
        print(f"{z.shape=}")
        for layer in self.encoder:             # TE layers
            z = layer(z)                       # returns 1 tensor in TE ≥2.4
        z = z[:, :T, :]                        # trim padding off  ← NEW
        z = z.flatten(1)                       # (B, T*d_model)   (T is 50)
        y = self.head(z).view(-1, 2, 40)       # (B, 2, 40)
        '''
        '''
        seq_aligned = x.size(1) % 16 == 0
        with te.fp8_autocast(enabled=seq_aligned, fp8_recipe=fp8_recipe):
            z = self.posenc(self.embed(x))      # (B, 50, d)
            for layer in self.encoder:
                z = layer(z)                 # TE layer returns (output, aux)
            z = z.flatten(1)                    # (B, 50*d_model)
            y = self.head(z).view(-1, 2, 40)
        '''
        
        '''
        z = self.posenc(self.embed(x))                        # (B, 50, d)
        z = self.encoder(z)                                   # 8-layer stack
        z = z.flatten(1)                                      # (B, 50*d_model)
        y = self.head(z).view(-1, 2, 40)                      # (B, 2, 40)
        '''
        return y

    
class ConvTransformer(nn.Module):
    """
    Maps (B, 50, 40) → (B, 2, 40)
      • circular Conv1d: 50 → 128 channels
      • Linear:          40 → 512 (d_model)
      • 8-layer Transformer encoder
    """
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        obs_dim: int = 40,
        in_timesteps: int = 50,
        conv_channels: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__()

        # 1. Temporal convolution (time-axis treated as channels)
        padding = kernel_size // 2
        self.temporal_conv = nn.Conv1d(
            in_channels=in_timesteps,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        # 2. Spatial embedding to d_model
        self.spatial_embed = nn.Linear(obs_dim, d_model)

        # 3. Positional encoding — your existing class, unmodified
        self.posenc = PositionalEncoding(d_model, max_len=conv_channels)

        # 4. Transformer encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

        # 5. Head → (B, 2, 40)
        self.head = nn.Linear(conv_channels * d_model, 80)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 50, 40)
        z = self.temporal_conv(x)            # (B, 128, 40)
        z = self.spatial_embed(z)            # (B, 128, 512)
        z = self.posenc(z)                   # + sinusoidal positions
        z = self.encoder(z)                  # Transformer stack
        z = z.flatten(1)                     # (B, 128*512)
        return self.head(z).view(-1, 2, 40)  # (B, 2, 40)


class _ResidualMLPBlock(nn.Module):
    """Pre-LN residual MLP block: h <- h + Linear(SiLU(LN(h)))."""
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, h: Tensor) -> Tensor:
        x = self.ln(h)
        x = F.silu(x)
        return h + self.fc(x)


class FCRecognition(nn.Module):
    r"""
    Fully-connected encoder for Lorenz-96 observations.

    Input:  y of shape (B, steps, L)
    Output: features of shape (B, out_channels, L)

    Two modes:
      - mode='global'       : one big MLP over (steps*L) -> (out_channels*L)
      - mode='per_location' : shared MLP over 'steps' for each of the L locations

    Args:
        steps:          number of observation steps (your previous 'steps')
        L:              spatial length (e.g., 40)
        out_channels:   output channels per location (default 2; set 1 if you want (B,1,L))
        hidden:         hidden width
        n_blocks:       number of residual MLP blocks (depth)
        mode:           'global' or 'per_location'
        dropout:        optional dropout applied after first projection (0 disables)
    """
    def __init__(
        self,
        steps: int,
        L: int,
        out_channels: int = 2,
        hidden: int = 512,
        n_blocks: int = 4,
        mode: Literal["global", "per_location"] = "global",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.steps = int(steps)
        self.L = int(L)
        self.out_channels = int(out_channels)
        self.hidden = int(hidden)
        self.n_blocks = int(n_blocks)
        self.mode = mode
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if mode == "global":
            in_dim = steps * L
            trunk = [nn.Linear(in_dim, hidden)]
            for _ in range(n_blocks):
                trunk.append(_ResidualMLPBlock(hidden))
            self.trunk = nn.Sequential(*trunk)
            self.head = nn.Linear(hidden, out_channels * L)
            # near-identity-ish start
            nn.init.zeros_(self.head.bias)

        elif mode == "per_location":
            in_dim = steps
            trunk = [nn.Linear(in_dim, hidden)]
            for _ in range(n_blocks):
                trunk.append(_ResidualMLPBlock(hidden))
            self.trunk = nn.Sequential(*trunk)
            self.head = nn.Linear(hidden, out_channels)
            nn.init.zeros_(self.head.bias)

        else:
            raise ValueError("mode must be 'global' or 'per_location'")

    def forward(self, y: Tensor) -> Tensor:
        """
        y: (B, steps, L)
        returns: (B, out_channels, L)
        """
        B, S, L = y.shape
        assert S == self.steps and L == self.L, f"Expected (B,{self.steps},{self.L}), got {tuple(y.shape)}"

        if self.mode == "global":
            # (B, S*L) -> trunk -> (B, out_channels*L) -> reshape (B, out_channels, L)
            x = y.reshape(B, S * L)
            h = self.trunk(x)
            h = self.do(h)
            out = self.head(h).view(B, self.out_channels, L)
            return out

        else:  # per_location
            # Shared MLP over steps for each location
            # (B, S, L) -> (B, L, S) -> (B*L, S) -> trunk -> (B*L, out_ch) -> (B, L, out_ch) -> (B, out_ch, L)
            x = y.permute(0, 2, 1).reshape(B * L, S)
            h = self.trunk(x)
            h = self.do(h)
            out = self.head(h).view(B, L, self.out_channels).permute(0, 2, 1)
            return out
