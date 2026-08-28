"""
Image Loader for RSNA Knee Abnormality Detection
Works with: data/train_series/StudyUID/SeriesUID/*.jpe
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
        try:
            img = Image.open(img_path).convert('L')
            return np.array(img, dtype=np.float32)
        except:
            return None
    
    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        if arr.max() == arr.min():
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - arr.min()) / (arr.max() - arr.min())).astype(np.float32)
    
    @staticmethod
    def resize(arr: np.ndarray, size: int = 224) -> np.ndarray:
        if arr.shape != (size, size):
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
        return arr.astype(np.float32)
    
    def load_study(self, study_uid: str, mode: str = 'train', 
                   target_size: int = 224, max_images: int = None):
        """Load all images for a study"""
        images = []
        
        study_dir = (self.train_dir if mode == 'train' else self.test_dir) / study_uid
        
        if not study_dir.exists():
            return None, {}
        
        # Load from all series subdirectories
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            
            # Get all image files
            for img_file in sorted(series_dir.glob('*.jpeg')) + sorted(series_dir.glob('*.jpg')):
                img = self.load_image(str(img_file))
                if img is not None:
                    img = self.normalize(img)
                    img = self.resize(img, target_size)
                    images.append(img)
        
        if not images:
            return None, {}
        
        images = np.array(images)
        
        if max_images and len(images) > max_images:
            idx = np.linspace(0, len(images)-1, max_images, dtype=int)
            images = images[idx]
        
        return images, {'num_images': len(images), 'shape': images.shape}
    
    def get_labels(self, study_uid: str):
        if self.train_csv is None:
            return {l: 0.5 for l in self.label_cols}
        
        try:
            row = self.train_csv[self.train_csv['StudyInstanceUID'] == study_uid].iloc[0]
            return {l: float(row[l]) if pd.notna(row[l]) else 0.5 for l in self.label_cols}
        except:
            return {l: 0.5 for l in self.label_cols}