"""
PyTorch Dataset for RSNA Knee Abnormality Detection (JPEG Dataset)
Works with pre-processed 224x224 JPEG images
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from image_loader import ImageLoader


class KneeAbnormalityJPEGDataset(Dataset):
    """
    PyTorch Dataset for knee abnormality detection (JPEG version)
    Loads study images on-the-fly
    """
    
    def __init__(self, csv_path: str, data_root: str, mode: str = 'train',
                 max_images: int = 64, augment: bool = False):
        """
        Initialize dataset
        
        Args:
            csv_path: Path to train.csv or test.csv
            data_root: Root directory with train_images/, test_images/, etc.
            mode: 'train' or 'test'
            max_images: Maximum images per study (default 64)
            augment: Whether to apply data augmentation (only for training)
        """
        self.mode = mode
        self.data_root = Path(data_root)
        self.max_images = max_images
        self.augment = augment and (mode == 'train')
        
        # Load metadata
        self.df = pd.read_csv(csv_path)
        self.study_uids = self.df['StudyInstanceUID'].values
        
        # Initialize loader
        self.loader = ImageLoader(str(data_root))
        
        # Define label columns
        self.label_cols = [
            'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
            'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
            'Synovitis', "Baker's", 'Contusion', 'Fracture'
        ]
        
        print(f"✓ Dataset initialized ({mode})")
        print(f"  Studies: {len(self)}")
        print(f"  Max images per study: {max_images}")
        print(f"  Augmentation: {self.augment}")
    
    def __len__(self):
        return len(self.study_uids)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a study (with all its images)
        
        Args:
            idx: Index in dataset
            
        Returns:
            Dictionary with:
                - 'study_uid': Study ID
                - 'images': Tensor of shape (max_images, H, W)
                - 'labels': Tensor of shape (12,)
                - 'num_images': Integer, actual number of images
        """
        study_uid = self.study_uids[idx]
        
        # Load all images for this study
        images, metadata = self.loader.load_study(
            study_uid,
            mode=self.mode,
            max_slices=None  # Load all, we'll trim later
        )
        
        if images is None:
            # Fallback: return zeros if loading fails
            images = np.zeros((self.max_images, 224, 224))
            labels = torch.tensor([0.5] * 12, dtype=torch.float32)
            return {
                'study_uid': study_uid,
                'images': torch.from_numpy(images).float(),
                'labels': labels,
                'num_images': 0
            }
        
        num_actual_images = len(images)
        
        # Apply augmentation if training (simple augmentation for JPEG)
        if self.augment:
            images = self._augment_images(images)
        
        # Pad or truncate to max_images
        if len(images) < self.max_images:
            pad_size = self.max_images - len(images)
            images = np.pad(
                images,
                ((0, pad_size), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
        else:
            # Evenly sample max_images slices
            indices = np.linspace(0, len(images) - 1, self.max_images, dtype=int)
            images = images[indices]
        
        # Convert to tensor
        images_tensor = torch.from_numpy(images).float()  # (max_images, H, W)
        
        # Get labels
        labels_dict = self.loader.get_labels(study_uid, mode=self.mode)
        labels = torch.tensor([
            labels_dict.get(col, 0.5)
            for col in self.label_cols
        ], dtype=torch.float32)
        
        return {
            'study_uid': study_uid,
            'images': images_tensor,  # (max_images, H, W)
            'labels': labels,          # (12,)
            'num_images': num_actual_images
        }
    
    def _augment_images(self, images: np.ndarray) -> np.ndarray:
        """Simple augmentation: random flips and rotation"""
        augmented = []
        
        for img in images:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            
            # Random brightness adjustment
            if np.random.rand() > 0.7:
                brightness_factor = np.random.uniform(0.8, 1.2)
                img = np.clip(img * brightness_factor, 0, 1)
            
            augmented.append(img)
        
        return np.array(augmented)


class DataLoaderCollator:
    """Custom collate function for DataLoader"""
    
    def __call__(self, batch: list) -> dict:
        """Collate batch of studies"""
        study_uids = [item['study_uid'] for item in batch]
        images = torch.stack([item['images'] for item in batch])  # (B, max_images, H, W)
        labels = torch.stack([item['labels'] for item in batch])   # (B, 12)
        num_images = torch.tensor([item['num_images'] for item in batch])
        
        return {
            'study_uids': study_uids,
            'images': images,
            'labels': labels,
            'num_images': num_images
        }


if __name__ == "__main__":
    # Test dataset
    print("Testing KneeAbnormalityJPEGDataset...\n")
    
    dataset = KneeAbnormalityJPEGDataset(
        csv_path='data/train.csv',
        data_root='data/',
        mode='train',
        max_images=64,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Study UID: {sample['study_uid']}")
    print(f"  Images shape: {sample['images'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Num actual images: {sample['num_images']}")
    
    # Test DataLoader
    print(f"\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    
    collator = DataLoaderCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collator
    )
    
    batch = next(iter(dataloader))
    print(f"Batch from DataLoader:")
    print(f"  Batch size: {len(batch['study_uids'])}")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    
    print(f"\n✓ Dataset tests passed!")
