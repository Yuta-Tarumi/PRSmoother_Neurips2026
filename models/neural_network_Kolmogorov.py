import math
from typing import List, Optional
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- helpers ------------------------------------------------------ #
def cpad2d(k: int):
    """Conv2d kwargs for ‘same’ padding on square kernels."""
    return dict(kernel_size=k, padding=k // 2, padding_mode="circular")


class _ResBlock2d(nn.Module):
    """Conv2d → Norm → ReLU → Conv2d → Norm (+ skip)"""

    def __init__(self, c_in: int, c_out: int, k: int = 3):
        super().__init__()
        self.use_skip = c_in == c_out

        self.conv1 = nn.Conv2d(c_in, c_out, **cpad2d(k))
        self.norm1 = nn.GroupNorm(1, c_out)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out, c_out, **cpad2d(k))
        self.norm2 = nn.GroupNorm(1, c_out)
        self.act2  = nn.ReLU(inplace=True)

        if not self.use_skip:               # channel change ⇒ linear proj
            self.skip_proj = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        skip = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.use_skip:
            x = x + skip                    # residual connection
        else:
            x = x + self.skip_proj(skip)    # channel-matching projection
        return self.act2(x)


# ---------- main network -------------------------------------------------- #
class Conv2DRecognition(nn.Module):
    """
    Periodic 2-D convolution encoder for Kolmogorov-flow observations.

    Channel progression (default time-stacked input with 50 frames):
        50 → 128 → 128 → 128 → 128 → 128 → 32 → 8 → 2

    Expected input shape  : *(B, 50, H, W)*  (H=W=256 in your data)
    Output feature shape : *(B, 2,  H, W)*   (final list entry)
    """

    CHANNELS: List[int] = [50, 64, 64, 32, 8, 2]

    def __init__(self, kernel_size: int = 5, n_frames: int = 50):
        """
        Args
        ----
        kernel_size : receptive field of every Conv2d (default 3×3)
        n_frames    : number of time slices stacked as channels (default 50)
        """
        super().__init__()

        # adapt first entry if user passes a different n_frames
        ch = self.CHANNELS.copy()
        ch[0] = n_frames

        layers = []
        for c_in, c_out in zip(ch[:-1], ch[1:]):
            # Residual block for every layer **except** the last one
            if c_out != ch[-1]:
                layers.append(_ResBlock2d(c_in, c_out, k=kernel_size))
            else:
                layers.append(nn.Conv2d(c_in, c_out, **cpad2d(kernel_size)))

        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y : Tensor  (B, n_frames, H, W)

        Returns
        -------
        Tensor     (B, final_channels, H, W)
        """
        return self.net(y)

# ---- helpers --------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Conv-patchify full-res inputs: (B,C,128,128) or (B,C,64,128) -> (B,N,D)"""
    def __init__(self, in_ch=10, embed_dim=512, ps=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=ps, stride=ps)
        self.ps = ps

    def forward(self, x):                                # x: (B, C, H, W), H,W divisible by ps
        x = self.proj(x)                                # -> (B, D, H/ps, W/ps)
        x = rearrange(x, "b d h w -> b (h w) d")        # -> (B, N, D) row-major (h then w)
        return x

class PatchUnEmbed(nn.Module):
    """Unpatchify tokens to image: (B,N,c*ps*ps) -> (B,c,128,128)"""
    def __init__(self, ps=16):
        super().__init__()
        self.ps = ps

    def forward(self, x, H=128, W=128):                 # x: (B, N, c*ps*ps)
        ps = self.ps
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                      h=H//ps, w=W//ps, p1=ps, p2=ps)
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):                                # (B, N, D)
        z = self.norm1(x)
        x = x + self.attn(z, z, z)[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------------- model ----------------------------

class ViT2D(nn.Module):
    """
    Inputs:
      - (B, C, 128, 128): full observation -> 64 tokens (ps=16)
      - (B, C,  64, 128): top half observed -> 32 observed + 32 masked -> 64 tokens
      - (B, C,  16,  16): coarse obs -> 1x1 conv -> avgpool to 8x8 -> 64 tokens
    Output:
      - (B, out_ch, 128, 128)
    """
    def __init__(self, ps=16, in_ch=10, out_ch=2, embed_dim=1024,
                 depth=8, n_heads=8, mlp_ratio=2., drop=0.0):
        super().__init__()
        assert 128 % ps == 0, "ps must divide 128"
        self.ps      = ps
        self.dim     = embed_dim
        self.out_ch  = out_ch

        # Full-res patch embed (stride ps)
        self.embed   = PatchEmbed(in_ch, embed_dim, ps)

        GH = GW = 128 // ps                             # e.g., 8 for ps=16
        self.N  = GH * GW
        self.pos = nn.Parameter(torch.randn(1, self.N, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Coarse 16x16 pathway: 1x1 conv -> avgpool to GHxGW -> tokens ----
        assert 16 % GH == 0, (
            f"For 16x16 inputs, need GH={GH} to divide 16. "
            "Choose ps so that 128/ps divides 16 (ps ∈ {8,16,32,64})."
        )
        ks = 16 // GH                                   # e.g., 2 when ps=16 (GH=8)
        self.coarse_proj_first = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        self.coarse_pool = nn.AvgPool2d(kernel_size=ks, stride=ks)

        # Transformer trunk
        self.blocks  = nn.Sequential(
            *[ViTBlock(embed_dim, n_heads, mlp_ratio, drop) for _ in range(depth)]
        )
        # Token head -> patch pixels
        self.head    = nn.Linear(embed_dim, out_ch * ps * ps)
        self.unfold  = PatchUnEmbed(ps)

    # ----------------------- token constructors -----------------------

    def _tokens_from_top_half(self, x_half: torch.Tensor) -> torch.Tensor:
        """
        x_half: (B, C, 64, 128)
        Returns full-grid tokens (B, N, D): top rows observed, bottom rows masked.
        """
        B = x_half.size(0)
        GH = GW = 128 // self.ps
        Nobs = (64 // self.ps) * GW                     # e.g., 4*8 = 32 when ps=16

        t_obs = self.embed(x_half)                      # (B, Nobs, D)
        mask_tok = self.mask_token.to(t_obs.dtype)
        tokens = mask_tok.expand(B, self.N, self.dim).clone()
        tokens[:, :Nobs, :] = t_obs                     # fill top rows in row-major order
        return tokens                                   # (B, N, D)

    def _tokens_from_coarse16(self, x_coarse: torch.Tensor) -> torch.Tensor:
        """
        x_coarse: (B, C, 16, 16)
        1x1 conv to mix channels -> AvgPool2d to GHxGW -> flatten to (B,N,D).
        """
        x = self.coarse_proj_first(x_coarse)            # (B, D, 16, 16)
        x = self.coarse_pool(x)                         # (B, D, GH, GW)
        x = rearrange(x, "b d h w -> b (h w) d")        # (B, N, D)
        return x

    # ----------------------------- forward -----------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,128,128) or (B,C,64,128) or (B,C,16,16)
        returns y: (B, out_ch, 128, 128)
        """
        _, _, H, W = x.shape

        if (H, W) == (128, 128):
            tokens = self.embed(x)                      # (B, N, D)
        elif (H, W) == (64, 128):
            tokens = self._tokens_from_top_half(x)      # (B, N, D)
        elif (H, W) == (16, 16):
            tokens = self._tokens_from_coarse16(x)      # (B, N, D)
        else:
            raise AssertionError(
                "Expected (B,C,128,128), (B,C,64,128), or (B,C,16,16)."
            )

        tokens = tokens + self.pos.to(tokens.dtype)     # add positional encodings
        tokens = self.blocks(tokens)
        y = self.head(tokens)                           # (B, N, out_ch*ps^2)
        y = self.unfold(y, H=128, W=128)                # (B, out_ch, 128, 128)
        return y
'''
class ViT2D_for_mixed(nn.Module):
    """
    Inputs:
      x_sparse: (B, T=10, 128,128), NaNs outside observations
        - HR: fully observed at [:,:64,:64]
        - LR: samples at [:, :, 3::8, 3::8] (16x16 grid)
    Pipeline (ps=8):
      - HR patchify 64x64 -> 8x8 -> 64 tokens
      - LR 16x16 -> 1x1 conv -> AvgPool(4) -> 4x4 -> 16 tokens   # <<< CHANGED (comment)
      - concat -> 80 tokens -> ViT                               # <<< CHANGED (comment)
      - reassemble to 16x16 tokens -> head -> unpatchify -> (B,out_ch,128,128)
    """
    def __init__(self, ps=8, in_ch=10, out_ch=2, embed_dim=1024,
                 depth=8, n_heads=8, mlp_ratio=2., drop=0.0,
                 use_type_embed=True):
        super().__init__()
        assert 128 % ps == 0 and ps == 8, "This variant assumes ps=8"

        self.ps = ps
        self.D  = embed_dim
        self.out_ch = out_ch
        self.use_type_embed = use_type_embed

        # HR patch embed (works on 64x64 -> 8x8 -> 64 tokens)
        self.embed_hr = PatchEmbed(in_ch=in_ch, embed_dim=embed_dim, ps=ps)

        # LR path: 16x16 -> 1x1 conv -> AvgPool(4) -> 4x4 -> 16 tokens
        self.lr_proj_1x1 = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        self.lr_pool_to_4 = nn.AvgPool2d(kernel_size=4, stride=4)   # <<< CHANGED (name + k,stride)

        # Positional embeddings for the mixed "sentence"
        # Keep separate grids for clarity then concat.
        self.pos_lr4 = nn.Parameter(torch.randn(1, 16, embed_dim))  # <<< CHANGED (name + shape 16)
        self.pos_hr8 = nn.Parameter(torch.randn(1, 64, embed_dim))  # (8x8) unchanged

        # Optional token-type embedding: 0=LR, 1=HR
        if self.use_type_embed:
            self.type_embed = nn.Embedding(2, embed_dim)

        # ViT trunk
        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, n_heads, mlp_ratio, drop) for _ in range(depth)]
        )

        # Decoder head to pixels (applied after reassembling 16x16 tokens)
        self.head   = nn.Linear(embed_dim, out_ch * ps * ps)
        self.unfold = PatchUnEmbed(ps=ps)

    # ---- token builders ----
    def _hr_tokens(self, x_hr64):  # (B,T,64,64)
        x_hr64 = torch.nan_to_num(x_hr64)
        t = self.embed_hr(x_hr64)          # (B, 64, D) for ps=8 on 64x64
        return t

    def _lr_tokens(self, x_lr16):  # (B,T,16,16)
        x_lr16 = torch.nan_to_num(x_lr16)
        x = self.lr_proj_1x1(x_lr16)       # (B, D,16,16)
        x = self.lr_pool_to_4(x)           # (B, D, 4, 4)                 # <<< CHANGED (name, output size)
        t = rearrange(x, "b d h w -> b (h w) d")  # (B, 16, D)            # <<< CHANGED (comment: 16 tokens)
        return t

    # ---- forward on sparse input ----
    def forward(self, x_sparse: torch.Tensor,
                hr_box=(0,0,64,64),
                lr_offset=3, lr_stride=8):
        """
        x_sparse: (B, T, 128,128)
        """
        B, T, H, W = x_sparse.shape
        assert (H, W) == (128, 128), "Expected (B,T,128,128)"

        # Extract HR (top-left quarter by default)
        y0, x0, hh, ww = hr_box
        assert (hh, ww) == (64,64), "Quarter HR assumed 64x64"
        x_hr = x_sparse[:, :, y0:y0+hh, x0:x0+ww]          # (B,T,64,64)

        # Extract LR 16x16
        x_lr = x_sparse[:, :, lr_offset::lr_stride, lr_offset::lr_stride]  # (B,T,16,16)
        assert x_lr.shape[-2:] == (16,16), "LR sampling must yield 16x16"

        # Build tokens
        t_lr = self._lr_tokens(x_lr)                       # (B, 16, D)     # <<< CHANGED (comment)
        t_hr = self._hr_tokens(x_hr)                       # (B, 64, D)

        # Add positional + (optional) type embeddings
        t_lr = t_lr + self.pos_lr4.to(t_lr.dtype)          # <<< CHANGED (pos name)
        t_hr = t_hr + self.pos_hr8.to(t_hr.dtype)
        if self.use_type_embed:
            tid_lr = torch.zeros((B, 16), dtype=torch.long, device=t_lr.device)  # <<< CHANGED (16)
            tid_hr = torch.ones((B, 64),  dtype=torch.long, device=t_lr.device)
            t_lr = t_lr + self.type_embed(tid_lr)
            t_hr = t_hr + self.type_embed(tid_hr)

        # 80-token sentence (16 LR + 64 HR)
        tokens_80 = torch.cat([t_lr, t_hr], dim=1)         # (B, 80, D)     # <<< CHANGED (name + comment)

        # Transformer
        z = self.blocks(tokens_80)                         # (B, 80, D)     # <<< CHANGED (var name)

        # Split back
        z_lr, z_hr = z[:, :16, :], z[:, 16:, :]            # (B,16,D), (B,64,D)  # <<< CHANGED (indices)

        # Reassemble a 16x16 token canvas for decoding
        z_lr_4x4 = rearrange(z_lr, "b (h w) d -> b h w d", h=4, w=4)         # <<< CHANGED (4x4)
        z_canvas = repeat(z_lr_4x4, "b h w d -> b (h r1) (w r2) d", r1=4, r2=4)  # (B,16,16,D)  # <<< CHANGED (r1=r2=4)

        z_hr_8x8 = rearrange(z_hr, "b (h w) d -> b h w d", h=8, w=8)  # (B,8,8,D)
        # overwrite top-left 8x8 with HR features
        z_canvas[:, :8, :8, :] = z_hr_8x8

        tokens_256 = rearrange(z_canvas, "b h w d -> b (h w) d")       # (B,256,D)

        # Decode to pixels
        y_tokens = self.head(tokens_256)                               # (B,256,out_ch*64)
        y = self.unfold(y_tokens, H=128, W=128)                        # (B,out_ch,128,128)
        return y
'''
class ViT2D_for_mixed(nn.Module):
    """
    Inputs:
      x_sparse: (B, T=10, 128,128), NaNs outside observations
        - HR: fully observed at [:,:64,:64]
        - LR: samples at [:, :, 3::8, 3::8] (16x16 grid)
    Pipeline (ps=8):
      - HR patchify 64x64 -> 8x8 -> 64 tokens
      - LR 16x16 -> 1x1 conv -> AvgPool(2) -> 8x8 -> 64 tokens
      - concat -> 128 tokens -> ViT
      - reassemble to 16x16 tokens -> head -> unpatchify -> (B,out_ch,128,128)
    """
    def __init__(self, ps=8, in_ch=10, out_ch=2, embed_dim=640,
                 depth=8, n_heads=8, mlp_ratio=2., drop=0.0,
                 use_type_embed=True):
        super().__init__()
        assert 128 % ps == 0 and ps == 8, "This variant assumes ps=8"

        self.ps = ps
        self.D  = embed_dim
        self.out_ch = out_ch
        self.use_type_embed = use_type_embed

        # HR patch embed (works on 64x64 -> 8x8 -> 64 tokens)
        self.embed_hr = PatchEmbed(in_ch=in_ch, embed_dim=embed_dim, ps=ps)

        # LR path: 16x16 -> 1x1 conv -> AvgPool(2) -> 8x8 -> 64 tokens
        self.lr_proj_1x1 = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        self.lr_pool_to_8 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Positional embeddings for the 128-token "sentence"
        # We keep separate grids for clarity then concat.
        self.pos_lr8 = nn.Parameter(torch.randn(1, 64, embed_dim))  # (8x8)
        self.pos_hr8 = nn.Parameter(torch.randn(1, 64, embed_dim))  # (8x8)

        # Optional token-type embedding: 0=LR, 1=HR
        if self.use_type_embed:
            self.type_embed = nn.Embedding(2, embed_dim)

        # ViT trunk
        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, n_heads, mlp_ratio, drop) for _ in range(depth)]
        )

        # Decoder head to pixels (applied after reassembling 16x16 tokens)
        self.head   = nn.Linear(embed_dim, out_ch * ps * ps)
        self.unfold = PatchUnEmbed(ps=ps)

    # ---- token builders ----
    def _hr_tokens(self, x_hr64):  # (B,T,64,64)
        # Replace any unexpected NaNs with 0
        x_hr64 = torch.nan_to_num(x_hr64)
        t = self.embed_hr(x_hr64)          # (B, 64, D) for ps=8 on 64x64
        return t

    def _lr_tokens(self, x_lr16):  # (B,T,16,16)
        x_lr16 = torch.nan_to_num(x_lr16)
        x = self.lr_proj_1x1(x_lr16)       # (B, D,16,16)
        x = self.lr_pool_to_8(x)           # (B, D, 8, 8)
        t = rearrange(x, "b d h w -> b (h w) d")  # (B, 64, D)
        return t

    # ---- forward on sparse input ----
    def forward(self, x_sparse: torch.Tensor,
                hr_box=(0,0,64,64),
                lr_offset=3, lr_stride=8):
        """
        x_sparse: (B, T, 128,128)
        """
        B, T, H, W = x_sparse.shape
        assert (H, W) == (128, 128), "Expected (B,T,128,128)"

        # Extract HR (top-left quarter by default)
        y0, x0, hh, ww = hr_box
        assert (hh, ww) == (64,64), "Quarter HR assumed 64x64"
        x_hr = x_sparse[:, :, y0:y0+hh, x0:x0+ww]          # (B,T,64,64)

        # Extract LR 16x16
        x_lr = x_sparse[:, :, lr_offset::lr_stride, lr_offset::lr_stride]  # (B,T,16,16)
        assert x_lr.shape[-2:] == (16,16), "LR sampling must yield 16x16"

        # Build tokens
        t_lr = self._lr_tokens(x_lr)                       # (B, 64, D)
        t_hr = self._hr_tokens(x_hr)                       # (B, 64, D)

        # Add positional + (optional) type embeddings
        t_lr = t_lr + self.pos_lr8.to(t_lr.dtype)
        t_hr = t_hr + self.pos_hr8.to(t_hr.dtype)
        if self.use_type_embed:
            tid_lr = torch.zeros((B, 64), dtype=torch.long, device=t_lr.device)
            tid_hr = torch.ones((B, 64),  dtype=torch.long, device=t_lr.device)
            t_lr = t_lr + self.type_embed(tid_lr)
            t_hr = t_hr + self.type_embed(tid_hr)

        # 128-token sentence
        tokens_128 = torch.cat([t_lr, t_hr], dim=1)        # (B, 128, D)

        # Transformer
        z = self.blocks(tokens_128)                        # (B, 128, D)

        # Split back
        z_lr, z_hr = z[:, :64, :], z[:, 64:, :]            # (B,64,D), (B,64,D)

        # Reassemble a 16x16 token canvas for decoding
        z_lr_8x8 = rearrange(z_lr, "b (h w) d -> b h w d", h=8, w=8)  # (B,8,8,D)
        z_canvas = repeat(z_lr_8x8, "b h w d -> b (h r1) (w r2) d", r1=2, r2=2)  # (B,16,16,D)

        z_hr_8x8 = rearrange(z_hr, "b (h w) d -> b h w d", h=8, w=8)  # (B,8,8,D)
        # overwrite top-left 8x8 with HR features
        z_canvas[:, :8, :8, :] = z_hr_8x8

        tokens_256 = rearrange(z_canvas, "b h w d -> b (h w) d")       # (B,256,D)

        # Decode to pixels
        y_tokens = self.head(tokens_256)                               # (B,256,out_ch*64)
        y = self.unfold(y_tokens, H=128, W=128)                        # (B,out_ch,128,128)
        return y

import torch
import torch.nn as nn
from einops import rearrange, repeat

class ViT2D_for_sparse(nn.Module):
    """
    Inputs:
      x_sparse: (B, T=10, 128, 128), NaNs outside observations
        - LR samples at [:, :, lr_offset::lr_stride, lr_offset::lr_stride] -> (16x16 grid)

    Pipeline (ps=8):
      - LR 16x16 -> 1x1 conv -> AvgPool(2) -> 8x8 -> 64 tokens
      - ViT over 64 tokens
      - upsample tokens 8x8 -> 16x16 by 2x2 repeat
      - head -> unpatchify -> (B, out_ch, 128, 128)
    """
    def __init__(self, ps=8, in_ch=10, out_ch=2, embed_dim=1024,
                 depth=8, n_heads=8, mlp_ratio=2., drop=0.0,
                 use_type_embed=False):
        super().__init__()
        assert 128 % ps == 0 and ps == 8, "This variant assumes ps=8"

        self.ps = ps
        self.D  = embed_dim
        self.out_ch = out_ch
        self.use_type_embed = use_type_embed

        # LR path: 16x16 -> 1x1 conv -> AvgPool(2) -> 8x8 -> 64 tokens
        self.lr_proj_1x1 = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        self.lr_pool_to_8 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Positional embeddings for the 64-token (8x8) "sentence"
        self.pos_lr8 = nn.Parameter(torch.randn(1, 64, embed_dim))  # (8x8)

        # Optional token-type embedding: (single type = LR)
        if self.use_type_embed:
            self.type_embed = nn.Embedding(1, embed_dim)

        # ViT trunk (expects 64 tokens)
        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, n_heads, mlp_ratio, drop) for _ in range(depth)]
        )

        # Decoder head to pixels (applied after reassembling 16x16 tokens)
        self.head   = nn.Linear(embed_dim, out_ch * ps * ps)
        self.unfold = PatchUnEmbed(ps=ps)

    def _lr_tokens(self, x_lr16):  # (B, T, 16, 16)
        x_lr16 = torch.nan_to_num(x_lr16)
        x = self.lr_proj_1x1(x_lr16)       # (B, D, 16, 16)
        x = self.lr_pool_to_8(x)           # (B, D,  8,  8)
        t = rearrange(x, "b d h w -> b (h w) d")  # (B, 64, D)
        return t

    def forward(self, x_sparse: torch.Tensor,
                lr_offset=3, lr_stride=8):
        """
        x_sparse: (B, T, 128, 128)
        """
        B, T, H, W = x_sparse.shape
        assert (H, W) == (128, 128), "Expected (B,T,128,128)"

        # Extract LR 16x16 samples from the full grid
        x_lr = x_sparse[:, :, lr_offset::lr_stride, lr_offset::lr_stride]  # (B,T,16,16)
        assert x_lr.shape[-2:] == (16, 16), "LR sampling must yield 16x16"

        # Build 64 LR tokens (8x8)
        t_lr = self._lr_tokens(x_lr)                       # (B, 64, D)

        # Add positional (and optional type) embeddings
        t_lr = t_lr + self.pos_lr8.to(t_lr.dtype)
        if self.use_type_embed:
            tid_lr = torch.zeros((B, 64), dtype=torch.long, device=t_lr.device)  # all LR
            t_lr = t_lr + self.type_embed(tid_lr)

        # Transformer over 64 tokens
        z = self.blocks(t_lr)                              # (B, 64, D)

        # Reassemble a 16x16 token canvas for decoding by 2x2 tiling
        z_8x8 = rearrange(z, "b (h w) d -> b h w d", h=8, w=8)   # (B,8,8,D)
        z_canvas = repeat(z_8x8, "b h w d -> b (h r1) (w r2) d", r1=2, r2=2)  # (B,16,16,D)
        tokens_256 = rearrange(z_canvas, "b h w d -> b (h w) d")  # (B,256,D)

        # Decode to pixels
        y_tokens = self.head(tokens_256)                          # (B,256,out_ch*64)
        y = self.unfold(y_tokens, H=128, W=128)                   # (B,out_ch,128,128)
        return y
