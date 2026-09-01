# src/train.py
"""
Training loop, validation, and checkpointing for Mask2Former traffic image segmentation.
Uses the model's own built-in matching loss over mask_labels/class_labels.

Only two checkpoint files are ever kept on disk: checkpoint_latest.pth (overwritten every
epoch, used to resume an interrupted run) and checkpoint_best.pth (overwritten only when
validation Dice improves, used for the final saved/published model) -- a full
Mask2Former-Large state dict is large enough that per-epoch files aren't worth the disk cost.
"""

import os
import logging
import datetime
import torch
from torch.optim import Adam
from src.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE, CHECKPOINT_DIR, ADE_MEAN, ADE_STD, RESULTS_CSV
from src.dataset import get_dataloader
from src.model.mask2former import CustomMask2Former
from src.evaluate import evaluate_model
from src.utils import set_seed, save_checkpoint, append_result_row

LATEST_CHECKPOINT_NAME = "checkpoint_latest.pth"
BEST_CHECKPOINT_NAME = "checkpoint_best.pth"


def train_model(
    train_images,
    train_masks,
    val_images,
    val_masks,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    log_interval=100,
    val_max_batches=None,
    resume_from=None,
    best_metric_key="mean_dice",
):
    """
    Trains Mask2Former for semantic segmentation, logs loss, validates each epoch, and
    checkpoints. val_max_batches=None (default) evaluates the full validation set every epoch;
    pass an int to only check that many batches (cheaper, less representative).

    resume_from: path to a checkpoint (typically CHECKPOINT_DIR/checkpoint_latest.pth) to
    resume from -- restores model + optimizer state and continues from the next epoch.

    best_metric_key: which key of the per-epoch validation metrics dict ("mean_dice",
    "mean_f1", or "mean_iou") determines what counts as "best" for checkpoint_best.pth.

    Returns (model, history) where model's weights are the BEST validation epoch seen, not
    necessarily the last epoch trained -- this is what train_model.py publishes.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    set_seed()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_path = os.path.join(CHECKPOINT_DIR, LATEST_CHECKPOINT_NAME)
    best_path = os.path.join(CHECKPOINT_DIR, BEST_CHECKPOINT_NAME)

    try:
        model = CustomMask2Former().to(DEVICE)
        # Frozen params are still in model.parameters(), but never receive/apply gradients.
        optimizer = Adam(model.parameters(), lr=learning_rate)

        start_epoch = 0
        best_metric_value = float('-inf')
        # run_id ties every row this training lineage writes to results.csv together, including
        # across resumes -- carried inside the checkpoint dict so no separate bookkeeping is needed.
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        if resume_from:
            checkpoint = torch.load(resume_from, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            run_id = checkpoint.get('run_id', run_id)
            logging.info(f"Resumed from {resume_from} at epoch {start_epoch} (run_id={run_id})")
            if os.path.isfile(best_path):
                best_checkpoint = torch.load(best_path, map_location=DEVICE)
                best_metric_value = best_checkpoint['val_metrics'][best_metric_key]
                logging.info(f"Best-so-far {best_metric_key}={best_metric_value:.4f} (from {best_path})")

        train_loader = get_dataloader(train_images, train_masks, BATCH_SIZE, ADE_MEAN, ADE_STD, shuffle=True, is_train=True)
        val_loader = get_dataloader(val_images, val_masks, BATCH_SIZE, ADE_MEAN, ADE_STD, shuffle=False, is_train=False)
    except Exception as e:
        logging.error(f"Failed to initialize training: {e}")
        raise RuntimeError(f"Failed to initialize training: {e}")

    loss_history = []
    val_metric_history = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0
        iter100_loss = None
        for idx, batch in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                outputs = model(
                    pixel_values=batch["pixel_values"].to(DEVICE),
                    mask_labels=[labels.to(DEVICE) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(DEVICE) for labels in batch["class_labels"]],
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                batch_size = batch["pixel_values"].size(0)
                running_loss += loss.item()
                num_samples += batch_size

                if idx % log_interval == 0 and idx > 0:
                    iter100_loss = running_loss / num_samples
                    logging.info(f"Iteration {idx} - loss: {iter100_loss}")
            except Exception as e:
                logging.error(f"Error during training batch {idx}: {e}")

        epoch_loss = running_loss / num_samples if num_samples > 0 else float('nan')
        loss_history.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        val_metrics = evaluate_model(model, val_loader, max_batches=val_max_batches)
        val_metric_history.append(val_metrics)
        logging.info(
            f"Validation Metrics - Mean Dice: {val_metrics['mean_dice']:.4f}, "
            f"Mean F1 (β=0.5): {val_metrics['mean_f1']:.4f}, Mean IoU: {val_metrics['mean_iou']:.4f}"
        )

        checkpoint_state = {
            'epoch': epoch + 1,
            'run_id': run_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'val_metrics': val_metrics,
        }
        save_checkpoint(checkpoint_state, latest_path)

        is_new_best = val_metrics[best_metric_key] > best_metric_value
        if is_new_best:
            best_metric_value = val_metrics[best_metric_key]
            save_checkpoint(checkpoint_state, best_path)
            logging.info(f"New best {best_metric_key}={best_metric_value:.4f} -> saved {best_path}")

        append_result_row({
            "run_id": run_id,
            "row_type": "epoch",
            "epoch": epoch + 1,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "iteration_100_loss": iter100_loss,
            "epoch_end_train_loss": epoch_loss,
            "mean_dice": val_metrics["mean_dice"],
            "mean_f1_beta0.5": val_metrics["mean_f1"],
            "mean_iou": val_metrics["mean_iou"],
            "new_best_checkpoint": is_new_best,
            "val_max_batches": val_max_batches if val_max_batches is not None else "full",
        }, RESULTS_CSV)

    if os.path.isfile(best_path):
        best_checkpoint = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        logging.info(
            f"Reloaded best checkpoint (epoch {best_checkpoint['epoch']}, "
            f"{best_metric_key}={best_checkpoint['val_metrics'][best_metric_key]:.4f}) for final save"
        )

    return model, {'loss': loss_history, 'val_metrics': val_metric_history}
