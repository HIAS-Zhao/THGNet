import argparse
import csv
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.archs import build_network  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark VSR inference FPS without image saving or metric computation."
    )
    parser.add_argument("--opt", required=True, help="Path to the yml option file.")
    parser.add_argument("--weights", default=None, help="Path to generator weights. Optional for pure speed tests.")
    parser.add_argument("--param_key", default="params", help="Checkpoint key, e.g. params or params_ema.")
    parser.add_argument("--strict_load", action="store_true", help="Use strict=True when loading weights.")
    parser.add_argument("--name", default=None, help="Method name written to the CSV.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu. FPS for the paper should use cuda.")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of LR frames in one tested sequence.")
    parser.add_argument("--height", type=int, default=160, help="LR input height for random-input mode.")
    parser.add_argument("--width", type=int, default=160, help="LR input width for random-input mode.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size. Use 1 for paper reporting.")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations before timing.")
    parser.add_argument("--runs", type=int, default=30, help="Timed iterations.")
    parser.add_argument("--data_root", default=None, help="Optional LR frame root. If omitted, random input is used.")
    parser.add_argument("--clip_name", default=None, help="Optional subfolder under data_root for real-frame mode.")
    parser.add_argument("--use_hfe", choices=["keep", "true", "false"], default="keep")
    parser.add_argument("--use_gloe", choices=["keep", "true", "false"], default="keep")
    parser.add_argument("--output", default="fps_results.csv", help="CSV path for appending results.")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_bool(network_opt, key, value):
    if value == "true":
        network_opt[key] = True
    elif value == "false":
        network_opt[key] = False


def load_state_dict_flexible(net, weights, param_key, strict):
    ckpt = torch.load(weights, map_location="cpu")
    if isinstance(ckpt, dict):
        if param_key in ckpt:
            state = ckpt[param_key]
        elif "params" in ckpt:
            state = ckpt["params"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    missing, unexpected = net.load_state_dict(cleaned, strict=strict)
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")


def make_random_input(batch, num_frames, height, width, device):
    return torch.rand(batch, num_frames, 3, height, width, device=device)


def read_image(path):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Real-frame mode requires opencv-python. Use random mode or install cv2.") from exc

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor


def make_real_input(data_root, clip_name, num_frames, device):
    root = Path(data_root)
    if clip_name:
        root = root / clip_name
    elif any(p.is_dir() for p in root.iterdir()):
        dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        root = dirs[0]

    frames = sorted([p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS])
    if len(frames) < num_frames:
        raise RuntimeError(f"{root} only has {len(frames)} frames, but --num_frames={num_frames}.")
    frames = frames[:num_frames]
    seq = torch.stack([read_image(p) for p in frames], dim=0).unsqueeze(0)
    return seq.to(device, non_blocking=True)


def cuda_sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(net, inp, warmup, runs, device):
    net.eval()
    torch.backends.cudnn.benchmark = True

    for _ in range(warmup):
        _ = net(inp)
    cuda_sync(device)

    start = time.perf_counter()
    for _ in range(runs):
        _ = net(inp)
    cuda_sync(device)
    elapsed = time.perf_counter() - start

    sec_per_seq = elapsed / runs
    fps = inp.shape[1] * runs / elapsed
    return sec_per_seq, fps


def append_csv(path, row):
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    opt = load_yaml(args.opt)
    network_opt = deepcopy(opt["network_g"])
    override_bool(network_opt, "use_hfe", args.use_hfe)
    override_bool(network_opt, "use_gloe", args.use_gloe)

    net = build_network(network_opt).to(device)
    if args.weights:
        load_state_dict_flexible(net, args.weights, args.param_key, args.strict_load)

    if args.data_root:
        inp = make_real_input(args.data_root, args.clip_name, args.num_frames, device)
    else:
        inp = make_random_input(args.batch, args.num_frames, args.height, args.width, device)

    sec_per_seq, fps = benchmark(net, inp, args.warmup, args.runs, device)
    method = args.name or network_opt.get("type", "unknown")
    params_m = sum(p.numel() for p in net.parameters()) / 1e6

    row = {
        "method": method,
        "weights": args.weights or "",
        "frames": inp.shape[1],
        "lr_size": f"{inp.shape[-2]}x{inp.shape[-1]}",
        "batch": inp.shape[0],
        "device": str(device),
        "params_m": f"{params_m:.4f}",
        "sec_per_sequence": f"{sec_per_seq:.6f}",
        "fps": f"{fps:.4f}",
        "use_hfe": str(network_opt.get("use_hfe", "NA")),
        "use_gloe": str(network_opt.get("use_gloe", "NA")),
    }
    append_csv(args.output, row)

    print("==== FPS benchmark ====")
    for key, value in row.items():
        print(f"{key}: {value}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
