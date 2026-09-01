# src/plot_results.py
"""
Renders a grid of training-metric plots (train loss, val/test Dice, F1, IoU vs. epoch) from
results/results.csv (see src.config.RESULTS_CSV / src.utils.append_result_row, which
src.train.train_model and evaluate_model.py write to automatically).

Standalone utility, not imported by the training/eval pipeline itself -- run manually after a
training run to (re)generate the plot for the README:

    python -m src.plot_results
    python -m src.plot_results --csv path/to/other_results.csv --run-id run_20260827_rtx4090
    python -m src.plot_results --out results/metrics_plot.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt

from src.config import RESULTS_CSV, RESULTS_DIR


def _read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_results(csv_path=RESULTS_CSV, run_id=None, out_path=None):
    """
    Reads csv_path (the append-only results log) and saves a 2x2 grid: train loss, val Dice,
    val F1, val IoU, each vs. epoch -- markers on epoch rows, the best epoch highlighted, and
    any "test" rows for the same run overlaid as a horizontal reference line.

    run_id: which run to plot. Defaults to the run_id of the LAST row in the file (i.e. the most
    recent training run/resume lineage) -- pass an explicit run_id to plot an older one instead.
    """
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows to plot")

    if run_id is None:
        run_id = rows[-1]["run_id"]

    epoch_rows = [r for r in rows if r["run_id"] == run_id and r["row_type"] == "epoch"]
    test_rows = [r for r in rows if r["run_id"] == run_id and r["row_type"] == "test"]
    if not epoch_rows:
        raise ValueError(f"No epoch rows found for run_id={run_id!r} in {csv_path}")

    epoch_rows.sort(key=lambda r: int(r["epoch"]))
    epochs = [int(r["epoch"]) for r in epoch_rows]
    train_loss = [_to_float(r["epoch_end_train_loss"]) for r in epoch_rows]
    val_dice = [_to_float(r["mean_dice"]) for r in epoch_rows]
    val_f1 = [_to_float(r["mean_f1_beta0.5"]) for r in epoch_rows]
    val_iou = [_to_float(r["mean_iou"]) for r in epoch_rows]

    best_idx = max(range(len(epoch_rows)), key=lambda i: (val_dice[i] if val_dice[i] is not None else float("-inf")))
    best_epoch = epochs[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Training Metrics vs. Epoch (best epoch: {best_epoch})")

    plot_specs = [
        (axes[0, 0], train_loss, "Train loss (epoch end)", "tab:red"),
        (axes[0, 1], val_dice, "Val Mean Dice", "tab:blue"),
        (axes[1, 0], val_f1, "Val Mean F1 (β=0.5)", "tab:green"),
        (axes[1, 1], val_iou, "Val Mean IoU", "tab:purple"),
    ]
    for ax, values, title, color in plot_specs:
        ax.plot(epochs, values, marker="o", color=color)
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best epoch ({best_epoch})")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for test_row in test_rows:
        test_dice = _to_float(test_row["mean_dice"])
        if test_dice is not None:
            axes[0, 1].axhline(test_dice, color="tab:orange", linestyle=":", linewidth=1,
                                label=f"test ({test_row.get('checkpoint_used', '')})")
            axes[0, 1].legend(fontsize=7)

    fig.tight_layout()

    out_path = out_path or os.path.join(RESULTS_DIR, "metrics_plot.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=RESULTS_CSV, help="Path to results.csv (default: %(default)s)")
    parser.add_argument("--run-id", default=None, help="Which run_id to plot (default: the most recent one in the file)")
    parser.add_argument("--out", default=None, help="Output PNG path (default: results/metrics_plot.png)")
    args = parser.parse_args()
    plot_results(csv_path=args.csv, run_id=args.run_id, out_path=args.out)
