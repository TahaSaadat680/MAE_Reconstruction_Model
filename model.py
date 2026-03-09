"""
MAE Model — architecture matching the actual mae_best.pth checkpoint.
Keys: encoder.* / decoder.* submodules.
Decoder uses nn.TransformerEncoder (standard PyTorch) layers.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class MAEConfig:
    image_size:  int   = 224
    patch_size:  int   = 16
    mask_ratio:  float = 0.75

    enc_dim:     int   = 768
    enc_depth:   int   = 12
    enc_heads:   int   = 12

    dec_dim:     int   = 384
    dec_depth:   int   = 12
    dec_heads:   int   = 6

    mlp_ratio:   float = 4.0
    drop:        float = 0.0


# ─── Building blocks (encoder) ────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)


class Attention(nn.Module):
    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv   = nn.Linear(dim, dim * 3, bias=True)
        self.proj  = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─── Encoder submodule ────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    """Produces keys: encoder.cls_token, encoder.pos_embed,
       encoder.patch_embed.*, encoder.blocks.*, encoder.norm.*"""

    def __init__(self, cfg: MAEConfig):
        super().__init__()
        n = (cfg.image_size // cfg.patch_size) ** 2

        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size,
                                      embed_dim=cfg.enc_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, cfg.enc_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n + 1, cfg.enc_dim))
        self.blocks      = nn.ModuleList([
            Block(cfg.enc_dim, cfg.enc_heads, cfg.mlp_ratio, cfg.drop)
            for _ in range(cfg.enc_depth)
        ])
        self.norm = nn.LayerNorm(cfg.enc_dim)

    def _random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        n_keep  = int(N * (1 - mask_ratio))
        noise   = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep    = ids_shuffle[:, :n_keep]

        x_masked = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0)   # 0 = visible, 1 = masked
        return x_masked, mask, ids_restore

    def forward(self, imgs, mask_ratio):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self._random_masking(x, mask_ratio)

        cls = self.cls_token + self.pos_embed[:, :1, :]
        x   = torch.cat([cls.expand(x.size(0), -1, -1), x], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore


# ─── Decoder submodule ────────────────────────────────────────────────────────

class _Decoder(nn.Module):
    """Produces keys: decoder.proj.*, decoder.mask_token, decoder.pos_embed,
       decoder.transformer.layers.*, decoder.norm.*, decoder.head.*"""

    def __init__(self, cfg: MAEConfig, n_patches: int):
        super().__init__()
        self.proj       = nn.Linear(cfg.enc_dim, cfg.dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches, cfg.dec_dim))  # no CLS in decoder

        # Standard PyTorch TransformerEncoder — gives keys
        # decoder.transformer.layers.X.self_attn.*, linear1/2, norm1/2
        dec_layer = nn.TransformerEncoderLayer(
            d_model=cfg.dec_dim,
            nhead=cfg.dec_heads,
            dim_feedforward=int(cfg.dec_dim * cfg.mlp_ratio),
            dropout=cfg.drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # pre-norm, matching MAE training convention
        )
        self.transformer = nn.TransformerEncoder(dec_layer,
                                                 num_layers=cfg.dec_depth,
                                                 norm=None)

        self.norm = nn.LayerNorm(cfg.dec_dim)
        self.head = nn.Linear(cfg.dec_dim, cfg.patch_size ** 2 * 3, bias=True)

    def forward(self, x, ids_restore):
        x = self.proj(x)                          # (B, N_enc+1, dec_dim)

        B, N_enc, D = x.shape
        n_patches   = ids_restore.size(1)            # 196
        n_visible   = N_enc - 1                      # drop CLS
        mask_tokens = self.mask_token.expand(B, n_patches - n_visible, -1)

        # Drop encoder CLS, append mask tokens, restore patch order
        x_full = torch.cat([x[:, 1:, :], mask_tokens], dim=1)   # (B, 196, D)
        x_full = x_full.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )                                            # (B, 196, D)
        x_full = x_full + self.pos_embed             # (B, 196, D) — no CLS

        x_full = self.transformer(x_full)
        x_full = self.norm(x_full)
        pred   = self.head(x_full)                   # (B, 196, patch_px)
        return pred


# ─── Full MAE Model ───────────────────────────────────────────────────────────

class MaskedAutoEncoder(nn.Module):

    def __init__(self, cfg: MAEConfig = MAEConfig()):
        super().__init__()
        self.cfg = cfg
        n = (cfg.image_size // cfg.patch_size) ** 2

        self.encoder = _Encoder(cfg)
        self.decoder = _Decoder(cfg, n)

    # ── static helpers (called from app.py) ───────────────────────────────────

    @staticmethod
    def patchify(imgs, patch_size):
        """(B,3,H,W) → (B, N, patch_size²×3)"""
        B, C, H, W = imgs.shape
        h = w = H // patch_size
        x = imgs.reshape(B, C, h, patch_size, w, patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)          # B,h,w,p,p,C
        return x.reshape(B, h * w, patch_size * patch_size * C)

    @staticmethod
    def unpatchify(x, patch_size, image_size):
        """(B, N, patch_size²×3) → (B,3,H,W)"""
        h = w = image_size // patch_size
        B, N, _ = x.shape
        x = x.reshape(B, h, w, patch_size, patch_size, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, 3, image_size, image_size)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.cfg.mask_ratio

        latent, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent, ids_restore)

        # Normalised per-patch loss (for training; unused during inference)
        target = self.patchify(imgs, self.cfg.patch_size)
        mean   = target.mean(dim=-1, keepdim=True)
        std    = (target.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        target = (target - mean) / std

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask