# src/dataset.py
"""
Dataset loading and Mask2Former-format batching for traffic image segmentation.
"""

import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass
from typing import List

from transformers import Mask2FormerImageProcessor
from src.config import ADE_MEAN, ADE_STD, LABEL2ID

UNLABELED_ID = LABEL2ID['unlabeled']


class SegmentationTransform:
    """
    ToTensor + ADE-mean/std normalization, plus a single random horizontal flip applied
    identically to both the image and the mask (p=0.5) when is_train=True.
    """

    def __init__(self, mean=ADE_MEAN, std=ADE_STD, is_train=True):
        self.is_train = is_train
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image, mask):
        image = self.img_transform(image)
        mask = torch.from_numpy(np.array(mask)).long()
        if self.is_train and torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        # -1 is the "no polygon matched this pixel" sentinel from raw-data conversion.
        mask[mask == -1] = UNLABELED_ID
        return image, mask


@dataclass
class SegmentationDataInput:
    original_image: np.ndarray
    transformed_image: torch.Tensor
    original_segmentation_map: np.ndarray
    transformed_segmentation_map: torch.Tensor


class SemanticSegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation supporting both directory and in-memory (list) data sources.
    """
    def __init__(self, images, masks, mean=ADE_MEAN, std=ADE_STD, is_train=True, from_memory=False):
        self.from_memory = from_memory
        self.transform = SegmentationTransform(mean, std, is_train)
        try:
            if from_memory:
                assert len(images) == len(masks), "Number of images and masks must match."
                self.images = images
                self.masks = masks
            else:
                if not os.path.isdir(images) or not os.path.isdir(masks):
                    logging.error(f"Image or mask directory not found: {images}, {masks}")
                    raise FileNotFoundError(f"Image or mask directory not found: {images}, {masks}")
                self.images_dir = images
                self.masks_dir = masks
                self.images = sorted([f for f in os.listdir(images) if f.endswith(('.png', '.jpg', '.jpeg'))])
                self.masks = sorted([f for f in os.listdir(masks) if f.endswith(('.png', '.jpg', '.jpeg'))])
                if len(self.images) != len(self.masks):
                    logging.error(f"Number of images and masks must match. Found {len(self.images)} images and {len(self.masks)} masks.")
                assert len(self.images) == len(self.masks), "Number of images and masks must match."
        except Exception as e:
            logging.error(f"Failed to initialize SemanticSegmentationDataset: {e}")
            raise RuntimeError(f"Failed to initialize SemanticSegmentationDataset: {e}")

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        # Without this, direct iteration (`for x in dataset`, used by prepare_data.py) falls
        # back to Python's legacy protocol of calling __getitem__(0), __getitem__(1), ... until
        # IndexError -- but __getitem__ wraps every exception into a RuntimeError, which isn't
        # recognized as "stop", so it crashes instead of finishing cleanly.
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx) -> SegmentationDataInput:
        try:
            if self.from_memory:
                image = self.images[idx]
                mask = self.masks[idx]
            else:
                img_path = os.path.join(self.images_dir, self.images[idx])
                mask_path = os.path.join(self.masks_dir, self.masks[idx])
                image = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path)
            transformed_image, transformed_mask = self.transform(image, mask)
            original_mask = np.array(mask)
            original_mask[original_mask == -1] = UNLABELED_ID
            return SegmentationDataInput(
                original_image=np.array(image),
                transformed_image=transformed_image,
                original_segmentation_map=original_mask,
                transformed_segmentation_map=transformed_mask
            )
        except Exception as e:
            logging.error(f"Failed to load sample {idx}: {e}")
            raise RuntimeError(f"Failed to load sample {idx}: {e}")


def build_preprocessor(ignore_index: int = 0) -> Mask2FormerImageProcessor:
    """
    Shared factory for the Mask2FormerImageProcessor, used both to build training/eval batches
    (Mask2FormerCollator, below) and to post-process model outputs (src.evaluate).
    """
    return Mask2FormerImageProcessor(
        ignore_index=ignore_index,
        do_reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )


class Mask2FormerCollator:
    """
    Batches a list of SegmentationDataInput into the format Mask2Former expects for training
    and evaluation: pixel_values, pixel_mask, mask_labels, class_labels, plus the untransformed
    original_images/original_segmentation_maps (needed for metric computation against
    full-resolution ground truth).

    ignore_index=0: pixels labeled class 0 ('road') are treated as the ignored/background class
    when building per-class mask and class labels, i.e. 'road' does not get its own
    predicted-mask query (road dominates most pixels in these scenes).
    """
    def __init__(self, ignore_index: int = 0):
        self.preprocessor = build_preprocessor(ignore_index)

    def __call__(self, batch: List[SegmentationDataInput]) -> dict:
        original_images = [sample.original_image for sample in batch]
        transformed_images = [sample.transformed_image for sample in batch]
        original_segmentation_maps = [sample.original_segmentation_map for sample in batch]
        transformed_segmentation_maps = [sample.transformed_segmentation_map for sample in batch]

        preprocessed_batch = self.preprocessor(
            transformed_images,
            segmentation_maps=transformed_segmentation_maps,
            return_tensors="pt",
        )
        preprocessed_batch["original_images"] = original_images
        preprocessed_batch["original_segmentation_maps"] = original_segmentation_maps
        return preprocessed_batch


def get_dataloader(images, masks, batch_size, mean=ADE_MEAN, std=ADE_STD, shuffle=True, is_train=True, from_memory=False):
    try:
        dataset = SemanticSegmentationDataset(images, masks, mean, std, is_train, from_memory=from_memory)
        collate_fn = Mask2FormerCollator()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    except Exception as e:
        logging.error(f"Failed to create dataloader: {e}")
        raise RuntimeError(f"Failed to create dataloader: {e}")
