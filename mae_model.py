"""
Masked Autoencoder (MAE) — ViT-Base backbone
Reconstructed from training notebook for inference deployment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ─── Building Blocks ──────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


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
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


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


# ─── MAE ──────────────────────────────────────────────────────────────────────

class MaskedAutoEncoder(nn.Module):

    def __init__(self, cfg: MAEConfig = MAEConfig()):
        super().__init__()
        self.cfg = cfg
        n  = (cfg.image_size // cfg.patch_size) ** 2
        px = cfg.patch_size ** 2 * 3  # pixels per patch

        # ── Encoder ──────────────────────────────────────────────────────────
        self.patch_embed  = PatchEmbed(cfg.image_size, cfg.patch_size,
                                       embed_dim=cfg.enc_dim)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, cfg.enc_dim))
        self.enc_pos_emb  = nn.Parameter(torch.zeros(1, n + 1, cfg.enc_dim))
        self.enc_blocks   = nn.ModuleList([
            Block(cfg.enc_dim, cfg.enc_heads, cfg.mlp_ratio, cfg.drop)
            for _ in range(cfg.enc_depth)
        ])
        self.enc_norm     = nn.LayerNorm(cfg.enc_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.enc_to_dec   = nn.Linear(cfg.enc_dim, cfg.dec_dim, bias=True)
        self.mask_token   = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
        self.dec_pos_emb  = nn.Parameter(torch.zeros(1, n + 1, cfg.dec_dim))
        self.dec_blocks   = nn.ModuleList([
            Block(cfg.dec_dim, cfg.dec_heads, cfg.mlp_ratio, cfg.drop)
            for _ in range(cfg.dec_depth)
        ])
        self.dec_norm     = nn.LayerNorm(cfg.dec_dim)
        self.dec_pred     = nn.Linear(cfg.dec_dim, px, bias=True)

        self._init_weights()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token,   std=0.02)
        nn.init.trunc_normal_(self.mask_token,  std=0.02)
        nn.init.trunc_normal_(self.enc_pos_emb, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_emb, std=0.02)

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

    def _random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        n_keep  = int(N * (1 - mask_ratio))
        noise   = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep    = ids_shuffle[:, :n_keep]

        x_masked = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0)  # 0 = visible, 1 = masked
        return x_masked, mask, ids_restore

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.cfg.mask_ratio

        # Encode
        x = self.patch_embed(imgs)
        x = x + self.enc_pos_emb[:, 1:, :]
        x, mask, ids_restore = self._random_masking(x, mask_ratio)
        cls = self.cls_token + self.enc_pos_emb[:, :1, :]
        x   = torch.cat([cls.expand(x.size(0), -1, -1), x], dim=1)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)

        # Decode
        x = self.enc_to_dec(x)
        B, N_enc, D = x.shape
        n_patches   = ids_restore.size(1)
        mask_tokens = self.mask_token.expand(B, n_patches - (N_enc - 1), -1)
        x_full = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_full = x_full.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_full = torch.cat([x[:, :1, :], x_full], dim=1)
        x_full = x_full + self.dec_pos_emb
        for blk in self.dec_blocks:
            x_full = blk(x_full)
        x_full = self.dec_norm(x_full)
        pred   = self.dec_pred(x_full[:, 1:, :])  # (B, N, px)

        # Loss (normalised per patch)
        target = self.patchify(imgs, self.cfg.patch_size)
        mean   = target.mean(dim=-1, keepdim=True)
        std    = (target.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        target = (target - mean) / std

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask