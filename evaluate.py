import sys
import time
import os
import re

import numpy as np

# We will generate a histogram plot for per-class accuracy.
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
from utils import FileLogger, test_GL_NP, test_network
from utils import set_loader, set_model, print_loader_info, print_dataset_info
from config.cli import parse_option

# NOTE:
# - Evaluation-only script. No training or optimization.
# - set_model(opt) is expected to auto-load checkpoints.


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _extract_epoch_from_ckpt_path(cp_load_path: str):
    """Extract epoch number from a checkpoint filename like 'ckpt_epoch_500.pth'.
    Returns int epoch if found, else None."""
    if not cp_load_path:
        return None
    fname = os.path.basename(cp_load_path)
    m = re.search(r"ckpt_epoch_(\d+)\.pth$", fname)
    return int(m.group(1)) if m else None


def _get_epoch_prefix(opt) -> str:
    """Build a filename prefix like 'epoch500_' from opt.cp_load_path.
    Returns '' if epoch can't be determined."""
    epoch = _extract_epoch_from_ckpt_path(getattr(opt, "cp_load_path", ""))
    return f"epoch{epoch}_" if epoch is not None else ""


def _save_per_class(per_class_acc: np.ndarray, save_dir: str, prefix: str = ""):
    """Save per-class accuracy as .npy/.csv, a histogram image, and threshold stats CSV.

    Files produced in save_dir (all with optional prefix):
      - {prefix}per_class_acc.npy
      - {prefix}per_class_acc.csv                  (columns: class_id,acc)
      - {prefix}per_class_acc_hist.png             (histogram of per-class accuracies, NaN ignored)
      - {prefix}per_class_acc_thresholds.csv       (<= 10% * N for N=1..10: count and indices)
    """
    _ensure_dir(save_dir)

    # Build paths with prefix
    npy_path = os.path.join(save_dir, f"{prefix}per_class_acc.npy")
    csv_path = os.path.join(save_dir, f"{prefix}per_class_acc.csv")
    hist_png_path = os.path.join(save_dir, f"{prefix}per_class_acc_hist.png")
    thr_csv_path = os.path.join(save_dir, f"{prefix}per_class_acc_thresholds.csv")

    # Save .npy (raw array, may contain NaN for absent classes)
    np.save(npy_path, per_class_acc)

    # Save CSV list of per-class acc
    with open(csv_path, "w") as f:
        f.write("class_id,acc\n")
        for cid, acc in enumerate(per_class_acc):
            if np.isnan(acc):
                f.write(f"{cid},nan\n")
            else:
                f.write(f"{cid},{acc:.6f}\n")

    # Generate histogram (ignore NaN) with large, paper-friendly fonts
    valid = per_class_acc[~np.isnan(per_class_acc)]

    # Use explicit Axes so we can control fonts and line widths precisely
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)  # high DPI for print clarity

    # Histogram
    bins = np.linspace(0, 100, 21)  # bin edges 0..100, step 5
    ax.hist(valid, bins=bins)

    # Big, clear fonts for a paper
    label_fs = 22   # axis label font size
    tick_fs = 20    # tick label font size

    ax.set_xlabel("Per-class accuracy (%)", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)

    # Thicker spines and grid for readability
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.grid(True, linestyle="--", alpha=0.5)

    # Tight layout and high-quality save
    fig.tight_layout()
    fig.savefig(hist_png_path, bbox_inches="tight")
    plt.close(fig)

    # Threshold stats: <= 10% * N, for N=1..10
    with open(thr_csv_path, "w") as f:
        f.write("threshold_percent,count,class_indices\n")
        for N in range(1, 11):
            thr = 10 * N  # threshold in percent
            mask_valid = ~np.isnan(per_class_acc)
            idx_valid = np.where(mask_valid)[0]
            acc_valid = per_class_acc[mask_valid]
            idx_under = idx_valid[acc_valid <= thr]
            # Join indices with semicolons to avoid CSV conflicts
            idx_str = ";".join(map(str, idx_under.tolist()))
            f.write(f"{thr},{len(idx_under)},{idx_str}\n")

    print(
        "Saved per-class artifacts:\n"
        f"  {npy_path}\n"
        f"  {csv_path}\n"
        f"  {hist_png_path}\n"
        f"  {thr_csv_path}"
    )


def evaluate(model, eval_labeled_train_loader, eval_unlabeled_train_loader, test_loader_eval, opt):
    """Run evaluation and return a dict with overall acc and per-class acc.

    Behavior:
      - Always uses Top-1 (k=1).
      - For sup_train_type == 'gl': runs test_GL_NP (NumPy Laplacian pipeline).
      - For sup_train_type == 'mlp': runs test_network with predictor='MLP'.
    """
    # Force Top-1 for this evaluation
    setattr(opt, "top", 1)

    results = {}

    if opt.sup_train_type == 'gl':
        overall, per_class = test_GL_NP(
            model,
            eval_labeled_train_loader,
            test_loader_eval,
            opt,
            unlabel_train_loader=eval_unlabeled_train_loader,
            return_per_class=True
        )
        results["test_acc"] = overall
        results["per_class_acc"] = per_class

    elif opt.sup_train_type == 'mlp':
        overall, per_class = test_network(
            model,
            eval_labeled_train_loader,   # safe to pass
            test_loader_eval,
            opt,
            predictor='MLP',
            return_per_class=True
        )
        results["test_acc"] = overall
        results["per_class_acc"] = per_class

    else:
        raise ValueError(f"Unsupported sup_train_type: {opt.sup_train_type}")

    return results


def _print_worst_k_classes(per_class_acc: np.ndarray, k: int = 5):
    """Print worst-k classes by accuracy, with class indices."""
    mask_valid = ~np.isnan(per_class_acc)
    valid_idx = np.where(mask_valid)[0]
    if valid_idx.size == 0:
        print("No valid per-class accuracy entries to rank (all NaN).")
        return
    valid_accs = per_class_acc[mask_valid]
    order = np.argsort(valid_accs)  # ascending
    k = min(k, order.size)
    worst_idx = valid_idx[order[:k]]
    worst_accs = per_class_acc[worst_idx]
    print(f"Worst {k} classes by Top-1 accuracy:")
    for rank, (cls_idx, acc) in enumerate(zip(worst_idx, worst_accs), 1):
        acc_str = "NaN" if np.isnan(acc) else f"{acc:.2f}%"
        print(f"  #{rank}: class_index={cls_idx}, acc={acc_str}")


def main(opt):
    # ---------------- Data loaders ----------------
    train_loaders, eval_loaders = set_loader(opt, augment_type=opt.augment_type)
    train_dataset_ss, train_loader_ss, unlabel_train_loader = train_loaders
    eval_labeled_train_loader, eval_unlabeled_train_loader, test_loader_eval = eval_loaders

    print("✓ Data loaders generated successfully.")
    print_dataset_info("train_dataset_ss", train_dataset_ss)
    print_loader_info("train_loader_ss", train_loader_ss)
    print_loader_info("unlabel_train_loader", unlabel_train_loader)
    print_loader_info("eval_labeled_train_loader", eval_labeled_train_loader)
    print_loader_info("eval_unlabeled_train_loader", eval_unlabeled_train_loader)
    print_loader_info("test_loader_eval", test_loader_eval)

    # ---------------- Model ----------------
    model = set_model(opt)

    # Parameter counts (informational only)
    m = model.module if hasattr(model, "module") else model
    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total params: {total_params} ({total_params/1e6:.3f}M)")
    print(f"Trainable params: {trainable_params} ({trainable_params/1e6:.3f}M)")

    # ---------------- Evaluation ----------------
    start = time.time()
    results = evaluate(
        model,
        eval_labeled_train_loader,
        eval_unlabeled_train_loader,
        test_loader_eval,
        opt
    )
    elapsed = time.time() - start

    print("===== Evaluation Results (Top-1) =====")
    overall = results.get("test_acc", None)
    if overall is not None:
        overall_pct = overall if overall > 1.0 else overall * 100.0
        print(f"Overall Accuracy: {overall_pct:.2f}%")

    per_class = results.get("per_class_acc", None)
    if per_class is not None:
        print(
            "Per-class Accuracy: "
            f"mean={np.nanmean(per_class):.2f}%, "
            f"min={np.nanmin(per_class):.2f}%, "
            f"max={np.nanmax(per_class):.2f}% "
            "(NaN for classes absent in the test set)"
        )
        _print_worst_k_classes(per_class, k=5)

    print(f"Elapsed time: {elapsed:.2f}s")

    # ---------------- Save artifacts ----------------
    if hasattr(opt, "save_folder") and opt.save_folder:
        _ensure_dir(opt.save_folder)

        # Build epoch-based prefix once
        prefix = _get_epoch_prefix(opt)

        # Save overall + per-class + elapsed together, with prefix
        record_path = os.path.join(opt.save_folder, f"{prefix}eval_results.npy")
        np.save(record_path, {"results": results, "elapsed_sec": elapsed})
        print(f"Saved overall results to: {record_path}")

        # Save per-class artifacts (npy/csv/hist/thresholds), with prefix
        if per_class is not None:
            _save_per_class(per_class, opt.save_folder, prefix=prefix)


if __name__ == '__main__':
    opt = parse_option()
    if hasattr(opt, "save_folder") and opt.save_folder:
        os.makedirs(opt.save_folder, exist_ok=True)

    # Compute prefix from checkpoint path for the log file as well
    _prefix = _get_epoch_prefix(opt)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if opt.save_folder:
        txt_path = os.path.join(opt.save_folder, f"{_prefix}eval_record_{timestamp}.txt")
    else:
        txt_path = f"{_prefix}eval_record_{timestamp}.txt"

    with open(txt_path, "w") as f:
        logger = FileLogger(f, sys.stdout)
        sys.stdout = logger
        try:
            if getattr(opt, "print_all_parameters", False):
                for key, value in vars(opt).items():
                    print(f"{key}: {value}")
            main(opt)
        finally:
            sys.stdout = sys.__stdout__
            print(f"Logs written to: {txt_path}")
