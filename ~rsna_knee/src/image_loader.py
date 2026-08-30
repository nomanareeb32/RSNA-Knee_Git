"""
Image Loader for RSNA Knee Abnormality Detection
Works with: data/train_series/StudyUID/SeriesUID/*.jpe (grayscale)
Converts to 3-channel RGB for ResNet50
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2


class ImageLoader:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.train_dir = self.data_root / 'train_series'
        self.test_dir = self.data_root / 'test_series'
        
        # Load CSVs if they exist
        train_csv_path = self.data_root / 'train.csv'
        if train_csv_path.exists():
            self.train_csv = pd.read_csv(train_csv_path)
        else:
            self.train_csv = None
        
        self.label_cols = [
            'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
            'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
            'Synovitis', "Baker's", 'Contusion', 'Fracture'
        ]
        
        print(f"✓ ImageLoader initialized")
        if self.train_csv is not None:
            print(f"  Train studies: {len(self.train_csv)}")
    
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """Load JPE/JPEG grayscale image and convert to RGB"""
        try:
            # Open image (already grayscale from .JPE format)
            img = Image.open(img_path)
            
            # Ensure it's grayscale
            if img.mode != 'L':
                img = img.convert('L')
            
            img_array = np.array(img, dtype=np.float32)
            
            # Convert grayscale (H, W) to 3-channel RGB (H, W, 3)
            # by repeating the single channel 3 times
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
            
            return img_array
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 using percentile clipping (medical imaging standard)"""
        # Use percentile clipping for better contrast in medical images
        p2, p98 = np.percentile(arr, [2, 98])
        arr = np.clip(arr, p2, p98)
        
        if arr.max() == arr.min():
            return np.zeros_like(arr, dtype=np.float32)
        
        normalized = (arr - arr.min()) / (arr.max() - arr.min())
        return normalized.astype(np.float32)
    
    @staticmethod
    def resize(arr: np.ndarray, size: int = 224) -> np.ndarray:
        """Resize image to target size"""
        if arr.shape[:2] != (size, size):
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
        return arr.astype(np.float32)
    
    def load_study(self, study_uid: str, mode: str = 'train', 
                   target_size: int = 224, max_images: int = None):
        """Load all images for a study from Study/Series/image.jpe structure"""
        images = []
        
        study_dir = (self.train_dir if mode == 'train' else self.test_dir) / study_uid
        
        if not study_dir.exists():
            return None, {}
        
        # Load from all series subdirectories
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            
            # Get all image files (.jpe, .jpeg, .jpg)
            image_files = (
                sorted(series_dir.glob('*.jpe')) + 
                sorted(series_dir.glob('*.JPE')) +
                sorted(series_dir.glob('*.jpeg')) +
                sorted(series_dir.glob('*.JPEG')) +
                sorted(series_dir.glob('*.jpg')) +
                sorted(series_dir.glob('*.JPG'))
            )
            
            for img_file in image_files:
                img = self.load_image(str(img_file))
                if img is not None:
                    img = self.normalize(img)
                    img = self.resize(img, target_size)
                    images.append(img)
        
        if not images:
            return None, {}
        
        images = np.array(images)  # (N, H, W, 3) - 3-channel RGB
        
        if max_images and len(images) > max_images:
            idx = np.linspace(0, len(images)-1, max_images, dtype=int)
            images = images[idx]
        
        return images, {'num_images': len(images), 'shape': images.shape}
    
    def get_labels(self, study_uid: str):
        """Get abnormality labels for a study"""
        if self.train_csv is None:
            return {l: 0.5 for l in self.label_cols}
        
        try:
            row = self.train_csv[self.train_csv['StudyInstanceUID'] == study_uid].iloc[0]
            return {l: float(row[l]) if pd.notna(row[l]) else 0.5 for l in self.label_cols}
        except:
            return {l: 0.5 for l in self.label_cols}