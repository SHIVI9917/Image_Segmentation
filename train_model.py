# train_model.py
"""
Script to train Mask2Former on traffic image segmentation data.
"""

import os
from src.train import train_model, LATEST_CHECKPOINT_NAME
from src.config import PROCESSED_DATA_DIR, MODEL_DIR, CHECKPOINT_DIR

def main():
    try:
        train_images = os.path.join(PROCESSED_DATA_DIR, "train", "images")
        train_masks = os.path.join(PROCESSED_DATA_DIR, "train", "masks")
        val_images = os.path.join(PROCESSED_DATA_DIR, "val", "images")
        val_masks = os.path.join(PROCESSED_DATA_DIR, "val", "masks")
        for d in [train_images, train_masks, val_images, val_masks]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Required directory not found: {d}")

        # Auto-resume from a leftover checkpoint (e.g. after an interruption) instead of restarting.
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, LATEST_CHECKPOINT_NAME)
        resume_from = latest_checkpoint if os.path.isfile(latest_checkpoint) else None
        if resume_from:
            print(f"Found existing checkpoint at {resume_from} -- resuming training from there.")

        model, history = train_model(train_images, train_masks, val_images, val_masks, resume_from=resume_from)

        final_model_dir = os.path.join(MODEL_DIR, "finetuned")
        os.makedirs(final_model_dir, exist_ok=True)
        model.model.save_pretrained(final_model_dir)
        print(f"Training complete. Final model saved to {final_model_dir}")
        print(f"Per-epoch train loss: {history['loss']}")
    except Exception as e:
        print(f"Error in train_model.py: {e}")

if __name__ == "__main__":
    main()
