import os
import sys
from typing import Tuple

import torch
from torchvision.utils import make_grid

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

# --- Ensure project root is on sys.path (robust import when running from anywhere) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import from your existing project (same as your training file) ---
from config.cli import parse_option
from utils import set_loader


# -----------------------------
# Default normalization presets
# -----------------------------
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD:  Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
CIFAR100_STD:  Tuple[float, float, float] = (0.2675, 0.2565, 0.2761)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def pick_mean_std_from_opt(opt) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Try to infer normalization stats from opt; otherwise fall back by dataset name,
    and finally default to CIFAR-10 if nothing matches.
    """
    # Prefer explicit mean/std on opt if available
    if hasattr(opt, "mean") and hasattr(opt, "std"):
        try:
            mean = tuple(float(x) for x in opt.mean)
            std = tuple(float(x) for x in opt.std)
            if len(mean) == 3 and len(std) == 3:
                return mean, std
        except Exception:
            pass

    # Otherwise use dataset name heuristics
    ds = getattr(opt, "dataset", "").lower()
    if "cifar100" in ds or "cifar-100" in ds:
        return CIFAR100_MEAN, CIFAR100_STD
    # Default to CIFAR-10
    return CIFAR10_MEAN, CIFAR10_STD


def unnormalize(imgs: torch.Tensor,
                mean: Tuple[float, float, float],
                std: Tuple[float, float, float]) -> torch.Tensor:
    """
    Reverse per-channel normalization and clamp to [0, 1].
    imgs: (B, 3, H, W) float tensor normalized by (imgs - mean) / std.
    """
    if imgs.dim() != 4 or imgs.size(1) != 3:
        raise ValueError("Expected imgs of shape (B, 3, H, W).")
    device = imgs.device
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    imgs = imgs * std_t + mean_t
    return imgs.clamp(0.0, 1.0)


def grid_to_numpy(grid_tensor: torch.Tensor):
    """Convert a (3, H, W) grid tensor in [0,1] to (H, W, 3) numpy array for plt.imshow."""
    if grid_tensor.dim() != 3 or grid_tensor.size(0) != 3:
        raise ValueError("Expected grid tensor of shape (3, H, W).")
    return grid_tensor.permute(1, 2, 0).cpu().numpy()


def save_two_views_figure(view1: torch.Tensor,
                          view2: torch.Tensor,
                          out_path: str,
                          n: int = 16,
                          nrow: int = 4,
                          mean: Tuple[float, float, float] = CIFAR10_MEAN,
                          std: Tuple[float, float, float] = CIFAR10_STD) -> None:
    """
    Save a side-by-side figure of two grids (first n images) to out_path.
    """
    # Select first n and detach
    n = min(n, view1.size(0), view2.size(0))
    v1 = view1[:n].detach()
    v2 = view2[:n].detach()

    # Unnormalize (keep on original device), then move to CPU for plotting
    v1 = unnormalize(v1, mean, std).cpu()
    v2 = unnormalize(v2, mean, std).cpu()

    # Make 4x4 (or nrow) grids
    grid1 = make_grid(v1, nrow=nrow, padding=2)  # (3, H', W')
    grid2 = make_grid(v2, nrow=nrow, padding=2)

    # Convert to numpy HWC
    g1 = grid_to_numpy(grid1)
    g2 = grid_to_numpy(grid2)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    axes[0].imshow(g1)
    axes[0].set_title(f"View 1 (first {n})")
    axes[0].axis("off")

    axes[1].imshow(g2)
    axes[1].set_title(f"View 2 (first {n})")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    # Parse options exactly like your training script
    opt = parse_option()

    # Build loader exactly like your training script (two views, strong aug, train split)
    _, train_loader = set_loader(
        opt,
        loader_suffix='Encoder Pretrain',
        augment_type='strong',
        twoviews=True,
        p_label=False,
        train=True,
        score_dataset=False
    )

    # Prepare output directory outside of opt.save_folder
    out_dir = os.path.join("save", "visualization")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "two_views_grid.png")

    # Pull one batch (images is expected to be a tuple/list of two tensors)
    images, labels = next(iter(train_loader))
    if not isinstance(images, (list, tuple)) or len(images) != 2:
        raise RuntimeError("Expected images to be a (view1, view2) pair when twoviews=True.")

    view1, view2 = images[0], images[1]

    # Infer mean/std from opt or dataset name for proper un-normalization
    mean, std = pick_mean_std_from_opt(opt)

    # Save side-by-side grids of the first 16 images
    save_two_views_figure(
        view1=view1,
        view2=view2,
        out_path=out_path,
        n=16,
        nrow=4,
        mean=mean,
        std=std
    )

    print(f"[OK] Saved two-view 4x4 grids to: {out_path}")


if __name__ == "__main__":
    main()
