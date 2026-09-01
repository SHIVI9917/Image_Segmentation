# src/evaluate.py
"""
Evaluation metrics for Mask2Former traffic image segmentation: per-class Dice, F1 (beta=0.5),
and IoU, averaged across classes, computed against the model's post-processed semantic
segmentation maps.
"""

import torch
import logging
from src.dataset import build_preprocessor
from src.config import DEVICE

_preprocessor = build_preprocessor()


def compute_dice_f1_iou_for_classes(pred_map, gt_map, beta=0.5, num_classes=None):
    """
    Computes the Dice coefficient, F1 score (beta), and IoU per class between a predicted and
    ground-truth segmentation map, then averages across classes.

    Note: classes that score exactly 0 are excluded from the mean rather than counted as 0 --
    this inflates the reported average relative to a strict per-class mean.
    """
    if num_classes is None:
        num_classes = max(pred_map.max().item(), gt_map.max().item()) + 1

    dice_scores, f1_scores, iou_scores = [], [], []

    for cls in range(num_classes):
        pred_cls_mask = (pred_map == cls).float()
        gt_cls_mask = (gt_map == cls).float()

        intersection = (pred_cls_mask * gt_cls_mask).sum()
        dice_coeff = (2 * intersection) / (pred_cls_mask.sum() + gt_cls_mask.sum() + 1e-8)

        precision = intersection / (pred_cls_mask.sum() + 1e-8)
        recall = intersection / (gt_cls_mask.sum() + 1e-8)
        f1_score = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)

        union = pred_cls_mask.sum() + gt_cls_mask.sum() - intersection
        iou = intersection / (union + 1e-8)

        if dice_coeff.item() > 0:
            dice_scores.append(dice_coeff.item())
        if f1_score.item() > 0:
            f1_scores.append(f1_score.item())
        if iou.item() > 0:
            iou_scores.append(iou.item())

    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    return mean_dice, mean_f1, mean_iou


def evaluate_model(model, dataloader, max_batches=None, device=DEVICE):
    """
    Evaluates a CustomMask2Former model on a dataloader (see src.dataset.get_dataloader),
    returning mean Dice, F1 (beta=0.5), and IoU averaged per-sample across the batches visited.

    Predictions are produced via post_process_semantic_segmentation against each sample's
    original (untransformed) resolution. Deliberately doesn't plot anything per sample -- that
    would break on a headless training environment (e.g. a RunPod pod with no display).
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    model.eval()
    running_dice = running_f1 = running_iou = 0.0
    num_samples = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if max_batches and idx >= max_batches:
                break
            try:
                pixel_values = batch["pixel_values"].to(device)
                outputs = model.predict(pixel_values)

                original_images = batch["original_images"]
                target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
                predicted_segmentation_maps = _preprocessor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )
                ground_truth_segmentation_maps = batch["original_segmentation_maps"]

                for pred_map, gt_map in zip(predicted_segmentation_maps, ground_truth_segmentation_maps):
                    pred_map = pred_map.clone().detach().to(device)
                    gt_map = torch.as_tensor(gt_map).to(device)

                    dice, f1, iou = compute_dice_f1_iou_for_classes(pred_map, gt_map, beta=0.5)
                    running_dice += dice
                    running_f1 += f1
                    running_iou += iou
                    num_samples += 1
            except Exception as e:
                logging.error(f"Error during evaluation batch {idx}: {e}")

    mean_dice = running_dice / num_samples if num_samples else float('nan')
    mean_f1 = running_f1 / num_samples if num_samples else float('nan')
    mean_iou = running_iou / num_samples if num_samples else float('nan')
    logging.info(f"Mean Dice: {mean_dice:.4f}, Mean F1 (beta=0.5): {mean_f1:.4f}, Mean IoU: {mean_iou:.4f}")
    return {"mean_dice": mean_dice, "mean_f1": mean_f1, "mean_iou": mean_iou}
