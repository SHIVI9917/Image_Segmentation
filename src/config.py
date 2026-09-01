# src/config.py
"""
Configuration file for Mask2Former Traffic Image Segmentation Project.
Contains paths, hyperparameters, and other global settings.
"""

import os
import torch

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model & output paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

# Append-only per-epoch/per-evaluation metrics log; see src/plot_results.py to render it.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'results.csv')


PRETRAINED_CHECKPOINT = "facebook/mask2former-swin-large-ade-semantic"

# IIT-H driving dataset, 40 classes. Mask2Former adds its own internal "no object" class on
# top of this at the model level.
ID2LABEL = {
    0: 'road', 1: 'parking', 2: 'drivable fallback', 3: 'sidewalk', 4: 'rail track',
    5: 'non-drivable fallback', 6: 'person', 7: 'animal', 8: 'rider', 9: 'motorcycle',
    10: 'bicycle', 11: 'autorickshaw', 12: 'car', 13: 'truck', 14: 'bus', 15: 'caravan',
    16: 'trailer', 17: 'train', 18: 'vehicle fallback', 19: 'curb', 20: 'wall', 21: 'fence',
    22: 'guard rail', 23: 'billboard', 24: 'traffic sign', 25: 'traffic light', 26: 'pole',
    27: 'polegroup', 28: 'obs-str-bar-fallback', 29: 'building', 30: 'bridge', 31: 'tunnel',
    32: 'vegetation', 33: 'sky', 34: 'fallback background', 35: 'unlabeled',
    36: 'ego vehicle', 37: 'rectification border', 38: 'out of roi', 39: 'license plate',
}
LABEL2ID = {label: id_ for id_, label in ID2LABEL.items()}

# Hyperparameters (update as needed)
NUM_CLASSES = len(ID2LABEL)
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
IMAGE_SIZE = (512, 512)
SEED = 42

# Normalization (ADE20K mean/std as in Mask2Former notebook)
ADE_MEAN = [123.675 / 255, 116.280 / 255, 103.530 / 255]
ADE_STD = [58.395 / 255, 57.120 / 255, 57.375 / 255]

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
