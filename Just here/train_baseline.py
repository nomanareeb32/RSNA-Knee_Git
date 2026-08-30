"""
train_baseline.py — RSNA Knee Abnormality Detection
Wires dataset_jpeg.py + image_loader.py + models.py + train.py together.

Folder layout expected:
    ~rsna_knee/
        data/
            train.csv
            train_series/{StudyUID}/{SeriesUID}/*.jpeg
            test_series/{StudyUID}/{SeriesUID}/*.jpeg
        src/
            dataset_jpeg.py
            image_loader.py
            models.py
            train.py
        config.yaml
        train_baseline.py   ← this file
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import yaml
import warnings
warnings.filterwarnings('ignore')

# ── src imports ──────────────────────────────────────────────────────────────
from image_loader import ImageLoader
from models import KneeAbnormalityModel
from train import Trainer

# ── Inline dataset (fixes batch key mismatch in dataset_jpeg.py) ─────────────

from torch.utils.data import Dataset

LABEL_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]


class KneeDataset(Dataset):
    """
    Loads JPEG studies and returns batches compatible with train.py.
    Key fix: returns 'slices' (not 'images') to match Trainer expectations.
    Images are expanded to 3-channel to satisfy ResNet input requirements.
    """

    def __init__(self, study_uids, data_root, mode='train',
                 max_slices=64, augment=False, image_size=224):
        self.study_uids = list(study_uids)
        self.loader = ImageLoader(data_root)
        self.mode = mode
        self.max_slices = max_slices
        self.augment = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, idx):
        study_uid = self.study_uids[idx]

        # Load images
        images, _ = self.loader.load_study(
            study_uid, mode=self.mode, max_slices=self.max_slices
        )

        if images is None:
            images = np.zeros((self.max_slices, self.image_size, self.image_size),
                              dtype=np.float32)
        else:
            images = self._pad_or_sample(images)

        # Augment
        if self.augment:
            images = self._augment(images)

        # (max_slices, H, W) → (max_slices, 3, H, W) for ResNet
        t = torch.from_numpy(images).float()          # (S, H, W)
        t = t.unsqueeze(1).expand(-1, 3, -1, -1)      # (S, 3, H, W)

        # Labels — use 0.5 for unlabeled studies (ignored in AUC calc)
        label_dict = self.loader.get_labels(study_uid, mode=self.mode)
        labels = torch.tensor(
            [label_dict[c] for c in LABEL_COLS], dtype=torch.float32
        )

        # Return dict with key 'slices' to match Trainer.train_epoch()
        return {
            'slices': t,           # (max_slices, 3, H, W)
            'labels': labels,      # (12,)
            'study_uid': study_uid
        }

    def _pad_or_sample(self, images):
        n = len(images)
        if n >= self.max_slices:
            idx = np.linspace(0, n - 1, self.max_slices, dtype=int)
            return images[idx].astype(np.float32)
        else:
            pad = np.zeros(
                (self.max_slices - n, images.shape[1], images.shape[2]),
                dtype=np.float32
            )
            return np.concatenate([images, pad], axis=0)

    def _augment(self, images):
        out = []
        for img in images:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() > 0.7:
                img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1)
            out.append(img)
        return np.array(out, dtype=np.float32)


def collate_fn(batch):
    """Stack dicts into a batched dict."""
    slices = torch.stack([b['slices'] for b in batch])   # (B, S, 3, H, W)
    labels = torch.stack([b['labels'] for b in batch])   # (B, 12)
    uids   = [b['study_uid'] for b in batch]
    return {'slices': slices, 'labels': labels, 'study_uids': uids}


# ─────────────────────────────────────────────────────────────────────────────
# Model forward-pass patch:
# models.py forward() expects (B, N, H, W) but we now pass (B, N, 3, H, W).
# Monkey-patch it here so we don't touch the original file.
# ─────────────────────────────────────────────────────────────────────────────
import types
from models import KneeAbnormalityModel as _KAM

def _patched_forward(self, slices):
    """
    Args:
        slices: (B, N, 3, H, W)  ← 3-channel JPEG input
    Returns:
        (B, 12)
    """
    B, N, C, H, W = slices.shape
    slices_flat = slices.reshape(B * N, C, H, W)          # (B*N, 3, H, W)
    slice_features = self.encoder.model(slices_flat)       # (B*N, feature_dim)
    slice_features = slice_features.reshape(B, N, -1)      # (B, N, feature_dim)
    return self.aggregator(slice_features)                 # (B, 12)

_KAM.forward = _patched_forward


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_root  = config['data']['data_root']    # 'data/'
    train_csv  = config['data']['train_csv']    # 'data/train.csv'
    max_slices = config['data']['max_slices']   # 64
    batch_size = config['training']['batch_size']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Discover available studies ───────────────────────────────────────────
    loader_probe = ImageLoader(data_root)
    all_uids_on_disk = loader_probe.get_all_study_uids(mode='train')
    print(f"\nStudies on disk: {len(all_uids_on_disk)}")

    if len(all_uids_on_disk) == 0:
        print("\n✗ No study folders found in data/train_series/")
        print("  Make sure the JPEG dataset finished downloading and unzipping.")
        print("  Run: kaggle datasets download -d alenic/rsna-knee-abnormality-detection-jpeg-224x224 -p data/ --unzip")
        return

    # ── Split: labeled studies stratified, unlabeled just shuffled ───────────
    df = pd.read_csv(train_csv)
    labeled_df = df.dropna(subset=LABEL_COLS, how='any')
    labeled_uids = set(labeled_df['StudyInstanceUID'].values)

    # Only use UIDs that are actually on disk
    available_labeled   = [u for u in all_uids_on_disk if u in labeled_uids]
    available_unlabeled = [u for u in all_uids_on_disk if u not in labeled_uids]

    print(f"Labeled studies on disk:   {len(available_labeled)}")
    print(f"Unlabeled studies on disk: {len(available_unlabeled)}")

    # Strategy: train on ALL studies (labeled + unlabeled),
    # validate AUC only on labeled studies
    val_frac = config['validation']['val_split']

    # Reserve some labeled studies for validation
    if len(available_labeled) >= 10:
        train_labeled, val_labeled = train_test_split(
            available_labeled, test_size=val_frac,
            random_state=42, shuffle=True
        )
    else:
        train_labeled = available_labeled
        val_labeled   = available_labeled  # too few to split

    train_uids = train_labeled + available_unlabeled
    val_uids   = val_labeled

    print(f"\nTrain set: {len(train_uids)} studies "
          f"({len(train_labeled)} labeled + {len(available_unlabeled)} unlabeled)")
    print(f"Val set:   {len(val_uids)} studies (all labeled)")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = KneeDataset(
        train_uids, data_root=data_root, mode='train',
        max_slices=max_slices, augment=True
    )
    val_ds = KneeDataset(
        val_uids, data_root=data_root, mode='train',   # labels live in train folder
        max_slices=max_slices, augment=False
    )

    num_workers = min(config['data']['num_workers'], 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'), persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'), persistent_workers=(num_workers > 0)
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = KneeAbnormalityModel(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        aggregation=config['model']['aggregation'],
        num_abnormalities=config['model']['num_abnormalities']
    )
    print(f"Parameters: {model.get_num_parameters():,}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, device, config['training'])

    # ── Go ───────────────────────────────────────────────────────────────────
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience']
    )

    print(f"\n✓ Done. Best Val AUC: {history['best_auc']:.4f} "
          f"(epoch {history['best_epoch'] + 1})")


if __name__ == '__main__':
    main()