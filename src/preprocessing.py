"""
Preprocessing and Data Augmentation for RSNA Knee Abnormality Detection
Optimized for RTX 4050
"""

import numpy as np
import cv2
from typing import Tuple
import albumentations as A


class DICOMPreprocessor:
    """Preprocessing pipeline for DICOM slices"""
    
    def __init__(self, target_size: int = 512, normalize_method: str = 'minmax'):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (default 512)
            normalize_method: 'minmax' or 'standardize'
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
    
    def normalize_intensity(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Normalize pixel intensities
        
        Args:
            pixel_array: Input array
            
        Returns:
            Normalized array
        """
        if self.normalize_method == 'minmax':
            # Clip to percentiles
            pmin, pmax = np.percentile(pixel_array, [2, 98])
            pixel_array = np.clip(pixel_array, pmin, pmax)
            
            # Normalize to 0-1
            if pmax > pmin:
                pixel_array = (pixel_array - pmin) / (pmax - pmin)
            else:
                pixel_array = np.zeros_like(pixel_array)
        
        elif self.normalize_method == 'standardize':
            # Z-score normalization
            mean = pixel_array.mean()
            std = pixel_array.std()
            if std > 0:
                pixel_array = (pixel_array - mean) / std
            else:
                pixel_array = np.zeros_like(pixel_array)
        
        return pixel_array.astype(np.float32)
    
    def resize_slice(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Resize to target size
        
        Args:
            pixel_array: Input array
            
        Returns:
            Resized array
        """
        if pixel_array.shape != (self.target_size, self.target_size):
            pixel_array = cv2.resize(
                pixel_array,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR
            )
        return pixel_array.astype(np.float32)
    
    def process_slice(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline
        
        Args:
            pixel_array: Input DICOM pixel array
            
        Returns:
            Processed slice
        """
        pixel_array = self.normalize_intensity(pixel_array)
        pixel_array = self.resize_slice(pixel_array)
        return pixel_array
    
    def augment_slice(self, pixel_array: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Data augmentation (only during training)
        
        Args:
            pixel_array: Input array (should be float32, 0-1 range)
            training: Whether to apply augmentation
            
        Returns:
            Augmented array
        """
        if not training:
            return pixel_array
        
        # Light augmentation suitable for medical images
        transform = A.Compose([
            A.Rotate(limit=15, interpolation=cv2.INTER_LINEAR, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(shear=(-10, 10), p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ], bbox_params=None)
        
        # Convert to uint8 for albumentations (expects 0-255)
        pixel_array_uint8 = (pixel_array * 255).astype(np.uint8)
        
        # Apply augmentation
        augmented = transform(image=pixel_array_uint8)
        pixel_array_aug = augmented['image']
        
        # Convert back to float32
        pixel_array_aug = pixel_array_aug.astype(np.float32) / 255.0
        
        return pixel_array_aug


class DataNormalizer:
    """Compute and apply dataset-wide normalization statistics"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.computed = False
    
    def compute_stats(self, data_iterator, num_samples: int = 100) -> Tuple[float, float]:
        """
        Compute mean and std from a subset of data
        
        Args:
            data_iterator: Iterator yielding image arrays
            num_samples: Number of samples to process
            
        Returns:
            Tuple of (mean, std)
        """
        means = []
        stds = []
        
        for i, img in enumerate(data_iterator):
            if i >= num_samples:
                break
            means.append(img.mean())
            stds.append(img.std())
        
        self.mean = np.mean(means)
        self.std = np.mean(stds)
        self.computed = True
        
        print(f"Computed statistics from {i+1} samples:")
        print(f"  Mean: {self.mean:.4f}")
        print(f"  Std: {self.std:.4f}")
        
        return self.mean, self.std
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using computed statistics
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if not self.computed:
            raise ValueError("Statistics not computed yet. Call compute_stats() first.")
        
        return (image - self.mean) / (self.std + 1e-7)
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse normalization
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        if not self.computed:
            raise ValueError("Statistics not computed yet. Call compute_stats() first.")
        
        return image * (self.std + 1e-7) + self.mean


if __name__ == "__main__":
    # Test preprocessing
    print("Testing DICOMPreprocessor...")
    
    preprocessor = DICOMPreprocessor(target_size=512)
    
    # Create dummy DICOM data
    dummy_dicom = np.random.randint(0, 4000, (960, 960), dtype=np.uint16)
    
    # Process
    processed = preprocessor.process_slice(dummy_dicom)
    print(f"Input shape: {dummy_dicom.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test augmentation
    augmented = preprocessor.augment_slice(processed, training=True)
    print(f"Augmented shape: {augmented.shape}")
    print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    print("\n✓ Preprocessing tests passed!")
