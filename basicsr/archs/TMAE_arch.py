import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/path/to/THGNet')
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.basicvsr_arch import ConvResidualBlocks


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y, weight=None):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if weight is not None:
            weight = weight.to(loss.dtype)
            if weight.shape[1] == 1 and loss.shape[1] != 1:
                weight = weight.expand(-1, loss.shape[1], -1, -1)
            weighted_sum = (loss * weight).sum()
            normalizer = weight.sum().clamp_min(1e-12)
            return weighted_sum / normalizer
        return loss.mean()


def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,H,W], return gradient magnitude approx [B,3,H,W]"""
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    _, c, _, _ = x.shape
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        return self.l1(sobel_grad(x), sobel_grad(y))


def make_block_mask(
    b: int,
    h: int,
    w: int,
    block: int = 16,
    mask_ratio: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    assert 0.0 < mask_ratio < 1.0
    assert h % block == 0 and w % block == 0, "H,W must be divisible by block size for clean block masking."

    gh, gw = h // block, w // block
    num_blocks = gh * gw
    k = int(round(num_blocks * mask_ratio))

    mask = torch.zeros((b, 1, gh, gw), device=device, dtype=torch.float32)
    for sample_idx in range(b):
        idx = torch.randperm(num_blocks, device=device)[:k]
        mask[sample_idx, 0].view(-1)[idx] = 1.0

    mask = mask.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
    return mask


class GatedTemporalFusion(nn.Module):
    """Center-guided gated fusion for temporal neighbor features."""

    def __init__(self, channels: int):
        super().__init__()
        self.center_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.neighbor_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, center_feat: torch.Tensor, neighbor_feats):
        center_base = self.center_proj(center_feat)
        aggregated = center_base
        for neighbor_feat in neighbor_feats:
            neighbor_base = self.neighbor_proj(neighbor_feat)
            gate = self.gate(torch.cat([center_base, neighbor_base], dim=1))
            aggregated = aggregated + gate * neighbor_base
        fused = self.out(torch.cat([center_base, aggregated], dim=1))
        return fused


@ARCH_REGISTRY.register()
class FeatureMAE_EncoderOnly(nn.Module):
    def __init__(self, MAE_path, mid_channels=64, enc_num_blocks=30, strict=True):
        super().__init__()
        self.encoder = ConvResidualBlocks(3, mid_channels, enc_num_blocks)
        ckpt = torch.load(MAE_path, map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=strict)

    def forward(self, x):
        return self.encoder(x)


@ARCH_REGISTRY.register()
class FeatureMAE_SR(nn.Module):
    def __init__(
        self,
        load_path: str = None,
        mid_channels: int = 64,
        enc_num_blocks: int = 30,
        block_size: int = 16,
        mask_ratio: float = 0.5,
        strict_load: bool = True,
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.block_size = block_size
        self.mask_ratio = mask_ratio

        self.encoder = ConvResidualBlocks(3, mid_channels, enc_num_blocks)
        self.dec_conv0 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.dec_up1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1)
        self.dec_up2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)
        self.dec_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.dec_out = nn.Conv2d(64, 3, 3, 1, 1)

        self.act = nn.LeakyReLU(0.1, inplace=True)
        if load_path:
            self.load_pretrained(load_path, strict=strict_load)

    def load_pretrained(self, load_path: str, strict: bool = True):
        ckpt = torch.load(load_path, map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=strict)
        print(f"[FeatureMAE_SR] Loaded best encoder weights from: {load_path}")

    @torch.no_grad()
    def encode_features(self, x_lr: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_lr)

    def forward(self, x_lr: torch.Tensor):
        feat = self.encoder(x_lr)
        b, _, h, w = feat.shape

        mask = make_block_mask(
            b, h, w,
            block=self.block_size,
            mask_ratio=self.mask_ratio,
            device=feat.device
        )

        feat_masked = feat * (1.0 - mask)

        x = self.act(self.dec_conv0(feat_masked))
        x = self.act(self.ps(self.dec_up1(x)))
        x = self.act(self.ps(self.dec_up2(x)))
        x = self.act(self.dec_hr(x))
        hr_hat = self.dec_out(x)

        return hr_hat, mask, feat


@ARCH_REGISTRY.register()
class TemporalNeighborFeatureMAE_SR(nn.Module):
    """
    Five-frame temporal-neighbor MAE for SR pretraining.

    The encoder always stays single-frame so it can be reused as the
    downstream feature extractor. Neighbor frames are partially masked to
    reduce answer leakage during pretraining.
    """

    def __init__(
        self,
        load_path: str = None,
        mid_channels: int = 64,
        enc_num_blocks: int = 30,
        block_size: int = 16,
        mask_ratio: float = 0.5,
        neighbor_mask_ratio: float = 0.3,
        num_frames: int = 5,
        strict_load: bool = True,
    ):
        super().__init__()
        assert num_frames >= 3 and num_frames % 2 == 1, "num_frames must be an odd number >= 3."
        assert 0.0 <= neighbor_mask_ratio < 1.0, "neighbor_mask_ratio must be in [0, 1)."

        self.mid_channels = mid_channels
        self.block_size = block_size
        self.mask_ratio = mask_ratio
        self.neighbor_mask_ratio = neighbor_mask_ratio
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        self.encoder = ConvResidualBlocks(3, mid_channels, enc_num_blocks)
        self.temporal_fusion = GatedTemporalFusion(mid_channels)

        self.dec_conv0 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.dec_up1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1)
        self.dec_up2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)
        self.dec_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.dec_out = nn.Conv2d(64, 3, 3, 1, 1)

        self.act = nn.LeakyReLU(0.1, inplace=True)
        if load_path:
            self.load_pretrained(load_path, strict=strict_load)

    def load_pretrained(self, load_path: str, strict: bool = True):
        ckpt = torch.load(load_path, map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=strict)
        print(f"[TemporalNeighborFeatureMAE_SR] Loaded encoder weights from: {load_path}")

    @torch.no_grad()
    def encode_features(self, x_lr: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_lr)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 5, "Expected input shape [B,T,3,H,W]."
        b, t, c, h, w = x_seq.shape
        assert t == self.num_frames, f"Expected T={self.num_frames}, but got {t}."

        feat_seq = self.encoder(x_seq.reshape(b * t, c, h, w))
        _, feat_c, feat_h, feat_w = feat_seq.shape
        feat_seq = feat_seq.reshape(b, t, feat_c, feat_h, feat_w)

        center_feat = feat_seq[:, self.center_idx, :, :, :]
        center_mask = make_block_mask(
            b,
            feat_h,
            feat_w,
            block=self.block_size,
            mask_ratio=self.mask_ratio,
            device=feat_seq.device
        )

        masked_center_feat = center_feat * (1.0 - center_mask)
        neighbor_feats = []
        for idx in range(t):
            if idx == self.center_idx:
                continue
            current_feat = feat_seq[:, idx, :, :, :]
            if self.neighbor_mask_ratio > 0:
                neighbor_mask = make_block_mask(
                    b,
                    feat_h,
                    feat_w,
                    block=self.block_size,
                    mask_ratio=self.neighbor_mask_ratio,
                    device=feat_seq.device
                )
                current_feat = current_feat * (1.0 - neighbor_mask)
            neighbor_feats.append(current_feat)

        fused_feat = self.temporal_fusion(masked_center_feat, neighbor_feats)

        x = self.act(self.dec_conv0(fused_feat))
        x = self.act(self.ps(self.dec_up1(x)))
        x = self.act(self.ps(self.dec_up2(x)))
        x = self.act(self.dec_hr(x))
        hr_hat = self.dec_out(x)

        center_mask_hr = F.interpolate(center_mask, scale_factor=4, mode="nearest")
        return hr_hat, center_mask, center_mask_hr, center_feat
