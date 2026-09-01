# src/model/mask2former.py
"""
Mask2Former model definition and utilities for traffic image segmentation.
Uses Hugging Face transformers for Mask2Former backbone.
"""

import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation
from src.config import PRETRAINED_CHECKPOINT, ID2LABEL


class CustomMask2Former(nn.Module):
    def __init__(self, id2label=None, pretrained_model=PRETRAINED_CHECKPOINT, freeze_pixel_level_module=True):
        super().__init__()
        id2label = id2label if id2label is not None else ID2LABEL
        label2id = {label: id_ for id_, label in id2label.items()}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        if freeze_pixel_level_module:
            self.freeze_pixel_level_module()

    def freeze_pixel_level_module(self):
        """
        Freezes the backbone and pixel decoder, leaving only the transformer decoder and MLP
        layer trainable, for resource-efficient finetuning.
        """
        for param in self.model.model.pixel_level_module.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, pixel_mask=None, mask_labels=None, class_labels=None):
        return self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

    def predict(self, pixel_values, pixel_mask=None):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs
