"""
Loads the RealWaste dataset, resizes images to 64x64, normalizes to [0, 1],
and produces a fixed, stratified 70/15/15 train/val/test split (seed=42).
Preprocessed arrays are cached to disk so later modules can just call
load_datasets() instead of re-reading and resizing every image.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 42
IMG_SIZE = 64

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "realwaste-main" / "RealWaste"
CACHE_PATH = ROOT / "data" / "realwaste_64.npz"

CLASS_NAMES = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
NUM_CLASSES = len(CLASS_NAMES)


def set_seed(seed=RANDOM_SEED):
    """Seed every random generator we use, so splits/results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_all_images():
    images = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = DATA_DIR / class_name
        for img_path in sorted(class_dir.glob("*.jpg")):
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(class_idx)
    return np.stack(images), np.array(labels, dtype=np.int64)


def _split_indices(labels):
    """Stratified 70/15/15 split: each class is split in the same proportions."""
    rng = np.random.default_rng(RANDOM_SEED)
    train_idx, val_idx, test_idx = [], [], []
    for class_idx in range(NUM_CLASSES):
        idx = np.where(labels == class_idx)[0]
        rng.shuffle(idx)
        n_train = int(round(len(idx) * 0.70))
        n_val = int(round(len(idx) * 0.15))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def load_datasets(force_reload=False):
    """Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    Images are NHWC float32 arrays in [0, 1]; labels are class indices into CLASS_NAMES.
    """
    set_seed()

    if CACHE_PATH.exists() and not force_reload:
        data = np.load(CACHE_PATH)
        return (
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"],
        )

    images, labels = _load_all_images()
    train_idx, val_idx, test_idx = _split_indices(labels)

    X_train, y_train = images[train_idx], labels[train_idx]
    X_val, y_val = images[val_idx], labels[val_idx]
    X_test, y_test = images[test_idx], labels[test_idx]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


class RealWasteDataset(Dataset):
    """Wraps preprocessed (NHWC float32) arrays as a PyTorch dataset of CHW tensors."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)  # HWC -> CHW
        label = int(self.labels[idx])
        return image, label


def get_dataloaders(batch_size=32):
    """Convenience: builds train/val/test DataLoaders straight from the cached split."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()
    set_seed()
    train_loader = DataLoader(RealWasteDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RealWasteDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(RealWasteDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()
    total = len(y_train) + len(y_val) + len(y_test)

    print(f"classes ({NUM_CLASSES}): {CLASS_NAMES}")
    print(f"train {len(y_train)} ({len(y_train) / total:.1%})")
    print(f"val   {len(y_val)} ({len(y_val) / total:.1%})")
    print(f"test  {len(y_test)} ({len(y_test) / total:.1%})")

    assert X_train.shape[1:] == (IMG_SIZE, IMG_SIZE, 3)
    assert X_train.dtype == np.float32
    assert 0.0 <= X_train.min() and X_train.max() <= 1.0
    assert set(np.unique(y_train)) <= set(range(NUM_CLASSES))
    print("OK")
