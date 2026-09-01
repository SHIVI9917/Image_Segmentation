# prepare_data.py
"""
Preprocess and split raw data for Mask2Former traffic image segmentation.

This script loads data from data/raw/data.pkl, splits it into train/val/test,
and saves processed images and masks to data/processed/.

Usage:
    python prepare_data.py
"""

import os
import pickle
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, ADE_MEAN, ADE_STD
from src.dataset import SemanticSegmentationDataset
import matplotlib.pyplot as plt

def main() -> None:
    """
    Main function to preprocess and split the dataset for Mask2Former training.
    Steps:
        1. Load data from data/raw/data.pkl.
        2. Split into train/val/test sets.
        3. Save processed images and masks to data/processed/.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    try:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        pkl_path = os.path.join(RAW_DATA_DIR, 'data.pkl')
        if not os.path.isfile(pkl_path):
            logging.error(f"data.pkl not found at {pkl_path}")
            raise FileNotFoundError(f"data.pkl not found at {pkl_path}")
        logging.info(f"Loading data from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data_set = pickle.load(f)

        images = []
        masks = []
        for idx, item in enumerate(data_set):
            img = item['pixel_values']
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img)
                except Exception as e:
                    logging.error(f"Sample {idx}: Could not convert pixel_values to PIL Image: {e}")
                    continue
            images.append(img)

            mask = np.array(item['label'])
            if not hasattr(mask, 'dtype') or not hasattr(mask, 'shape'):
                logging.error(f"Sample {idx}: label is not a numpy array.")
                continue
            if mask.ndim != 2:
                logging.error(f"Sample {idx}: label mask is not 2D, shape={mask.shape}")
                continue
            if not (np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.uint8)):
                try:
                    mask = mask.astype(np.int32)
                except Exception as e:
                    logging.error(f"Sample {idx}: Could not convert mask to int: {e}")
                    continue
            masks.append(mask)
            if idx%20: plt.imshow(mask, cmap='gray')
            

        if len(images) != len(masks):
            logging.error(f"Mismatch between images and masks after parsing: {len(images)} images, {len(masks)} masks.")
            raise ValueError("Mismatch between images and masks after parsing.")

        indices = list(range(len(images)))
        train_idx, valtest_idx = train_test_split(indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(valtest_idx, test_size=0.5, random_state=42)

        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }


        for split, idxs in splits.items():
            split_img_dir = os.path.join(PROCESSED_DATA_DIR, split, 'images')
            split_mask_dir = os.path.join(PROCESSED_DATA_DIR, split, 'masks')
            os.makedirs(split_img_dir, exist_ok=True)
            os.makedirs(split_mask_dir, exist_ok=True)

            valid_idxs = [i for i in idxs if i < len(images) and i < len(masks)]
            if len(valid_idxs) < len(idxs):
                logging.warning(f"Some indices in split '{split}' were out of range and have been removed.")

            dataset = SemanticSegmentationDataset(
                [images[i] for i in valid_idxs],
                [masks[i] for i in valid_idxs],
                ADE_MEAN, ADE_STD, is_train=(split=='train'), from_memory=True
            )

            for i, sample in enumerate(dataset):
                try:
                    img = Image.fromarray(sample.original_image).resize(IMAGE_SIZE)
                    mask = sample.original_segmentation_map
                    # Ensure mask is a PIL Image and resize with nearest neighbor
                    if isinstance(mask, Image.Image):
                        mask = mask.resize(IMAGE_SIZE, resample=Image.NEAREST)
                    else:
                        # Cast to uint8 before Image.fromarray: label ids are 0-39 (fits
                        # comfortably), and Image.fromarray on the raw int32 array would pick
                        # 32-bit "I" mode, which Pillow is deprecating for PNG output.
                        mask = Image.fromarray(mask.astype(np.uint8)).resize(IMAGE_SIZE, resample=Image.NEAREST)
                    img.save(os.path.join(split_img_dir, f"img_{i:05d}.png"))
                    mask.save(os.path.join(split_mask_dir, f"mask_{i:05d}.png"))
                except Exception as e:
                    logging.error(f"Error saving sample {i} in split {split}: {e}")
            logging.info(f"Saved {split} set: {len(dataset)} images to {split_img_dir} and masks to {split_mask_dir}")

        logging.info("Data preprocessing and splitting complete.")
    except Exception as e:
        logging.error(f"Error in prepare_data.py: {e}")

if __name__ == "__main__":
    main()
