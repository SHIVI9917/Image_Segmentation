# src/utils.py
"""
Utility functions for logging, visualization, and reproducibility.
"""

import os
import csv
import random
import numpy as np
import torch

# Fixed column order for results/results.csv. One row per training epoch (row_type="epoch")
# or per evaluation run (row_type="test") -- see append_result_row's docstring.
RESULTS_CSV_FIELDS = [
    "run_id", "row_type", "epoch", "timestamp",
    "iteration_100_loss", "epoch_end_train_loss",
    "mean_dice", "mean_f1_beta0.5", "mean_iou",
    "new_best_checkpoint", "checkpoint_used", "val_max_batches",
]


def append_result_row(row: dict, csv_path):
    """
    Appends one row to the append-only results CSV (creating it with a header if needed).
    `row` may omit any of RESULTS_CSV_FIELDS -- missing keys are written blank.

    row_type="epoch" (src.train.train_model, one per epoch): epoch, iteration_100_loss,
    epoch_end_train_loss, mean_dice/mean_f1_beta0.5/mean_iou, new_best_checkpoint,
    val_max_batches populated.

    row_type="test" (evaluate_model.py, one per evaluation run): mean_dice/mean_f1_beta0.5/
    mean_iou, checkpoint_used, epoch populated.
    """
    import logging
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULTS_CSV_FIELDS})
    logging.info(f"Appended {row.get('row_type', 'result')} row to {csv_path}")


def set_seed(seed=42):
    """
    Sets random seed for reproducibility across numpy, random, and torch.
    Logs the seed used.
    """
    import logging
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def save_checkpoint(state, filename):
    """
    Saves model state to a file and logs the action.
    """
    import logging
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def plot_sample(image, mask, pred_mask=None, class_names=None):
    """
    Visualizes an image, its ground truth mask, and optionally a predicted mask.
    Handles both numpy arrays and torch tensors robustly. Logs errors if visualization fails.
    """
    import logging
    try:
        import matplotlib.pyplot as plt
        n_plots = 3 if pred_mask is not None else 2
        plt.figure(figsize=(12, 4))
        plt.subplot(1, n_plots, 1)
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
        else:
            img = image
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, n_plots, 2)
        plt.imshow(mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask, cmap='jet', alpha=0.7)
        plt.title('Ground Truth')
        plt.axis('off')
        if pred_mask is not None:
            plt.subplot(1, n_plots, 3)
            plt.imshow(pred_mask.cpu().numpy() if isinstance(pred_mask, torch.Tensor) else pred_mask, cmap='jet', alpha=0.7)
            plt.title('Prediction')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Plotting failed: {e}")
