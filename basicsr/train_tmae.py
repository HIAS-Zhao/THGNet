import os
import glob
import time
import random
from dataclasses import dataclass
from collections import defaultdict
import sys
sys.path.insert(0, '/path/to/THGNet')
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from basicsr.archs.TMAE_arch import TemporalNeighborFeatureMAE_SR, CharbonnierLoss, GradLoss


def set_seed(seed: int = 3407, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def broadcast_scalar(value: float, device: torch.device, src: int = 0) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor.item()


class TemporalPairedDataset(Dataset):
    """Build 5-frame LR windows and supervise with the center-frame HR."""

    def __init__(
        self,
        lr_dir,
        hr_dir,
        scale=4,
        num_frames=5,
        augment=True,
        lr_crop_size=160,
        hr_crop_size=640,
        force_full_size=True,
        allowed_seqs=None,
    ):
        super().__init__()
        assert num_frames >= 3 and num_frames % 2 == 1, "num_frames must be an odd number >= 3."
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.num_frames = num_frames
        self.radius = num_frames // 2
        self.augment = augment
        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = hr_crop_size
        self.force_full_size = force_full_size
        self.allowed_seqs = set(allowed_seqs) if allowed_seqs is not None else None

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        hr_files = []
        for ext in exts:
            hr_files.extend(glob.glob(os.path.join(hr_dir, "**", ext), recursive=True))
        hr_files = [p for p in sorted(hr_files) if os.path.isfile(p)]

        seq_to_pairs = defaultdict(list)
        for hf in hr_files:
            rel = os.path.relpath(hf, hr_dir)
            seq = rel.split(os.sep)[0]
            if self.allowed_seqs is not None and seq not in self.allowed_seqs:
                continue
            lf = os.path.join(lr_dir, rel)
            if os.path.isfile(lf):
                seq_to_pairs[seq].append((lf, hf))

        self.samples = []
        for seq, pairs in sorted(seq_to_pairs.items()):
            pairs = sorted(pairs)
            if len(pairs) < num_frames:
                continue
            for center in range(self.radius, len(pairs) - self.radius):
                window = pairs[center - self.radius:center + self.radius + 1]
                lr_paths = [lr for lr, _ in window]
                center_hr_path = window[self.radius][1]
                self.samples.append((lr_paths, center_hr_path, seq))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No temporal samples found.\n"
                f"  HR dir: {hr_dir}\n  LR dir: {lr_dir}\n"
                f"Need at least {num_frames} mirrored frames per sequence folder."
            )

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, path: str) -> torch.Tensor:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Not a file: {path}")
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _random_crop(self, lr_seq: torch.Tensor, hr: torch.Tensor):
        _, _, h, w = lr_seq.shape
        _, H, W = hr.shape
        lr_cs = self.lr_crop_size
        hr_cs = self.hr_crop_size

        if h == lr_cs and w == lr_cs and H == hr_cs and W == hr_cs:
            return lr_seq, hr

        if h < lr_cs or w < lr_cs:
            raise ValueError(f"LR smaller than crop size: {(h, w)} vs {lr_cs}")
        if H < hr_cs or W < hr_cs:
            raise ValueError(f"HR smaller than crop size: {(H, W)} vs {hr_cs}")

        x0 = random.randint(0, w - lr_cs)
        y0 = random.randint(0, h - lr_cs)
        lr_crop = lr_seq[:, :, y0:y0 + lr_cs, x0:x0 + lr_cs]

        X0 = x0 * self.scale
        Y0 = y0 * self.scale
        hr_crop = hr[:, Y0:Y0 + hr_cs, X0:X0 + hr_cs]
        return lr_crop, hr_crop

    def _augment(self, lr_seq: torch.Tensor, hr: torch.Tensor):
        if random.random() < 0.5:
            lr_seq = torch.flip(lr_seq, dims=[3])
            hr = torch.flip(hr, dims=[2])
        if random.random() < 0.5:
            lr_seq = torch.flip(lr_seq, dims=[2])
            hr = torch.flip(hr, dims=[1])
        k = random.randint(0, 3)
        if k > 0:
            lr_seq = torch.rot90(lr_seq, k, dims=[2, 3])
            hr = torch.rot90(hr, k, dims=[1, 2])
        return lr_seq, hr

    def __getitem__(self, idx):
        lr_paths, hr_path, seq = self.samples[idx]
        lr_seq = torch.stack([self._read_rgb(path) for path in lr_paths], dim=0)
        hr = self._read_rgb(hr_path)

        if not self.force_full_size:
            lr_seq, hr = self._random_crop(lr_seq, hr)

        if self.augment:
            lr_seq, hr = self._augment(lr_seq, hr)

        return lr_seq, hr, seq


@torch.no_grad()
def calc_psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    pred = pred.clamp(0.0, 1.0)
    gt = gt.clamp(0.0, 1.0)
    mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean().item()


@torch.no_grad()
def validate_by_folder(model, val_loader, device, rank):
    if val_loader is None:
        return None, None

    model.eval()
    sum_psnr = defaultdict(float)
    cnt = defaultdict(int)

    iterator = tqdm(val_loader, desc="Val", leave=False) if is_main_process(rank) else val_loader
    for lr_seq, hr, seq in iterator:
        lr_seq = lr_seq.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        hr_hat, _, _, _ = model(lr_seq)
        psnr = calc_psnr(hr_hat, hr)

        seq_name = seq[0] if isinstance(seq, (list, tuple)) else str(seq)
        sum_psnr[seq_name] += psnr
        cnt[seq_name] += 1

    seq_psnr = {k: sum_psnr[k] / max(1, cnt[k]) for k in sum_psnr.keys()}
    mean_psnr = float(np.mean(list(seq_psnr.values()))) if len(seq_psnr) > 0 else -1e9
    return seq_psnr, mean_psnr


@dataclass
class TrainConfig:
    lr_dir: str = "/mnt/data0/VSR/SAT-MAT-VSR/train/LR4xBicubic"
    hr_dir: str = "/mnt/data0/VSR/SAT-MAT-VSR/train/GT"
    save_dir: str = "/home/wangyibo/RSVSR/DA5/basicsr/tmae_15_smv_pth"

    seed: int = 3407
    deterministic: bool = True

    val_ratio: float = 0.1

    mid_channels: int = 64
    enc_num_blocks: int = 30
    block_size: int = 16
    mask_ratio: float = 0.5
    neighbor_mask_ratio: float = 0.3
    num_frames: int = 5

    epochs: int = 200
    batch_size: int = 4
    num_workers: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.0

    lambda_grad: float = 0.1

    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4

    force_full_size: bool = True
    lr_crop_size: int = 160
    hr_crop_size: int = 640


def collect_all_seqs(hr_dir: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    hr_files = []
    for ext in exts:
        hr_files.extend(glob.glob(os.path.join(hr_dir, "**", ext), recursive=True))
    hr_files = [p for p in sorted(hr_files) if os.path.isfile(p)]
    return sorted({os.path.relpath(p, hr_dir).split(os.sep)[0] for p in hr_files})


def save_best_encoder(path: str, encoder_state: dict, epoch: int, best_psnr: float, cfg: TrainConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_mean_psnr": best_psnr,
        "encoder_state_dict": encoder_state,
        "seed": cfg.seed,
        "arch": {
            "type": "ConvResidualBlocks",
            "pretrain": "TemporalNeighborFeatureMAE_SR",
            "in_channels": 3,
            "out_channels": cfg.mid_channels,
            "num_blocks": cfg.enc_num_blocks,
            "num_frames": cfg.num_frames,
            "neighbor_mask_ratio": cfg.neighbor_mask_ratio,
        }
    }, path)


def main():
    cfg = TrainConfig()
    assert cfg.enc_num_blocks == 30, "enc_num_blocks must stay aligned with downstream feat_extract."
    assert cfg.num_frames == 5, "This training script is configured for 5-frame temporal-neighbor MAE."

    is_distributed, rank, world_size, local_rank = setup_distributed()
    set_seed(cfg.seed + rank, cfg.deterministic)
    os.makedirs(cfg.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
    else:
        device = torch.device("cpu")
    if is_main_process(rank):
        print("Device:", device)
        print(f"Distributed: {is_distributed} | World size: {world_size} | Rank: {rank}")

    all_seqs = collect_all_seqs(cfg.hr_dir)
    rng = random.Random(cfg.seed)
    rng.shuffle(all_seqs)

    n_val_seq = max(1, int(round(len(all_seqs) * cfg.val_ratio)))
    val_seqs = set(all_seqs[:n_val_seq])
    train_seqs = set(all_seqs[n_val_seq:])

    if is_main_process(rank):
        print(f"Total seq folders: {len(all_seqs)} | Train: {len(train_seqs)} | Val: {len(val_seqs)}")
        print("Val seqs (first 20):", sorted(list(val_seqs))[:20])

    train_set = TemporalPairedDataset(
        lr_dir=cfg.lr_dir,
        hr_dir=cfg.hr_dir,
        num_frames=cfg.num_frames,
        augment=True,
        force_full_size=cfg.force_full_size,
        lr_crop_size=cfg.lr_crop_size,
        hr_crop_size=cfg.hr_crop_size,
        allowed_seqs=train_seqs,
    )
    val_set = TemporalPairedDataset(
        lr_dir=cfg.lr_dir,
        hr_dir=cfg.hr_dir,
        num_frames=cfg.num_frames,
        augment=False,
        force_full_size=cfg.force_full_size,
        lr_crop_size=cfg.lr_crop_size,
        hr_crop_size=cfg.hr_crop_size,
        allowed_seqs=val_seqs,
    )

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None

    g = torch.Generator().manual_seed(cfg.seed + rank)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True
    )

    val_loader = None
    if is_main_process(rank):
        val_loader = DataLoader(
            val_set,
            batch_size=4,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfg.seed),
            drop_last=False
        )

    if is_main_process(rank):
        with open(os.path.join(cfg.save_dir, "split.txt"), "w", encoding="utf-8") as f:
            f.write("TRAIN_SEQS\n" + "\n".join(sorted(list(train_seqs))) + "\n\n")
            f.write("VAL_SEQS\n" + "\n".join(sorted(list(val_seqs))) + "\n")

    model = TemporalNeighborFeatureMAE_SR(
        mid_channels=cfg.mid_channels,
        enc_num_blocks=cfg.enc_num_blocks,
        block_size=cfg.block_size,
        mask_ratio=cfg.mask_ratio,
        neighbor_mask_ratio=cfg.neighbor_mask_ratio,
        num_frames=cfg.num_frames,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    charbonnier = CharbonnierLoss().to(device)
    grad_loss = GradLoss().to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)

    best_mean_psnr = -1e9
    best_epoch = -1
    best_path = os.path.join(cfg.save_dir, "best_encoder.pth")
    epochs_without_improvement = 0

    if is_main_process(rank):
        with open(os.path.join(cfg.save_dir, "config.txt"), "w", encoding="utf-8") as f:
            f.write(str(cfg) + "\n")
            f.write(f"distributed={is_distributed}, world_size={world_size}\n")

    try:
        for epoch in range(1, cfg.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            t0 = time.time()
            running_loss = 0.0

            iterator = tqdm(train_loader, desc=f"Train E{epoch:03d}", leave=False) if is_main_process(rank) else train_loader
            for lr_seq, hr, _seq in iterator:
                lr_seq = lr_seq.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)

                hr_hat, _center_mask, center_mask_hr, _feat = model(lr_seq)

                loss_rec = charbonnier(hr_hat, hr, weight=center_mask_hr)
                loss_g = grad_loss(hr_hat, hr)
                loss = loss_rec + cfg.lambda_grad * loss_g

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

                running_loss += loss.item()
                if is_main_process(rank):
                    iterator.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

            sched.step()
            train_loss = running_loss / max(1, len(train_loader))
            dt = time.time() - t0
            lr_now = optim.param_groups[0]["lr"]

            seq_psnr = None
            mean_psnr = None
            if is_main_process(rank):
                seq_psnr, mean_psnr = validate_by_folder(model, val_loader, device, rank)
                print(
                    f"[Epoch {epoch:03d}/{cfg.epochs}] "
                    f"train_loss={train_loss:.5f} "
                    f"val_mean_psnr(folder-avg)={mean_psnr:.3f} "
                    f"lr={lr_now:.2e} time={dt:.1f}s"
                )

                improved = mean_psnr > (best_mean_psnr + cfg.early_stop_min_delta)
                if improved:
                    best_mean_psnr = mean_psnr
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    save_best_encoder(
                        path=best_path,
                        encoder_state=unwrap_model(model).encoder.state_dict(),
                        epoch=epoch,
                        best_psnr=best_mean_psnr,
                        cfg=cfg
                    )

                    print(f"  New BEST! epoch={epoch}, best_mean_psnr={best_mean_psnr:.3f} -> saved {best_path}")
                    for seq_name in sorted(seq_psnr.keys()):
                        print(f"    ValSeq {seq_name}: PSNR={seq_psnr[seq_name]:.3f}")
                else:
                    epochs_without_improvement += 1
                    print(
                        f"  No improvement for {epochs_without_improvement} epoch(s). "
                        f"Patience={cfg.early_stop_patience}"
                    )

                print(f"  Current BEST: epoch={best_epoch}, best_mean_psnr={best_mean_psnr:.3f}")

                stop_flag = 1.0 if epochs_without_improvement >= cfg.early_stop_patience else 0.0
                if stop_flag > 0:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch={best_epoch}, best_mean_psnr={best_mean_psnr:.3f}"
                    )
            else:
                stop_flag = 0.0

            if is_distributed:
                stop_flag = broadcast_scalar(stop_flag, device)
                _ = broadcast_scalar(best_mean_psnr if is_main_process(rank) else 0.0, device)

            if stop_flag > 0:
                break

        if is_main_process(rank):
            print(f"Training done. Best epoch={best_epoch}, best folder-avg PSNR={best_mean_psnr:.3f}")
            print("Best encoder path:", best_path)
    finally:
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
