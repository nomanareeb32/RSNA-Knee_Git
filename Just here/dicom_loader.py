"""
DICOM Loader for RSNA Knee Abnormality Detection
Works with actual DICOM data structure:
data/train_series/StudyUID/SeriesUID/*.dcm
"""

import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import cv2
from tqdm import tqdm


class DICOMLoader:
    """Load and manage DICOM files from RSNA dataset"""
    
    def __init__(self, data_root: str):
        """
        Initialize loader with data root path
        
        Args:
            data_root: Root directory containing train_series/, train.csv, etc.
        """
        self.data_root = Path(data_root)
        
        # Load metadata
        self.train_csv = pd.read_csv(self.data_root / 'train.csv')
        self.train_series_csv = pd.read_csv(self.data_root / 'train_series.csv')
        self.test_csv = pd.read_csv(self.data_root / 'test.csv')
        self.test_series_csv = pd.read_csv(self.data_root / 'test_series.csv')
        
        # Define label columns
        self.label_cols = [
            'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
            'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
            'Synovitis', "Baker's", 'Contusion', 'Fracture'
        ]
        
        print(f"✓ Loaded metadata")
        print(f"  Train studies: {len(self.train_csv)}")
        print(f"  Test studies: {len(self.test_csv)}")
        print(f"  Train series: {len(self.train_series_csv)}")
        print(f"  Test series: {len(self.test_series_csv)}")
    
    @staticmethod
    def load_dicom_array(dcm_path: str) -> np.ndarray:
        """Load single DICOM file and return pixel array"""
        try:
            dcm = pydicom.dcmread(dcm_path)
            pixel_array = dcm.pixel_array.astype(np.float32)
            return pixel_array
        except Exception as e:
            print(f"Error loading {dcm_path}: {e}")
            return None
    
    @staticmethod
    def normalize_intensity(pixel_array: np.ndarray) -> np.ndarray:
        """Normalize pixel intensities to 0-1 range using percentile clipping"""
        pmin, pmax = np.percentile(pixel_array, [2, 98])
        pixel_array = np.clip(pixel_array, pmin, pmax)
        
        if pmax > pmin:
            pixel_array = (pixel_array - pmin) / (pmax - pmin)
        else:
            pixel_array = np.zeros_like(pixel_array)
        
        return pixel_array.astype(np.float32)
    
    @staticmethod
    def resize_slice(pixel_array: np.ndarray, target_size: int = 512) -> np.ndarray:
        """Resize slice to target size"""
        if pixel_array.shape != (target_size, target_size):
            pixel_array = cv2.resize(
                pixel_array,
                (target_size, target_size),
                interpolation=cv2.INTER_LINEAR
            )
        return pixel_array.astype(np.float32)
    
    def process_slice(self, dcm_path: str, target_size: int = 512) -> np.ndarray:
        """Full preprocessing pipeline for single slice"""
        pixel_array = self.load_dicom_array(dcm_path)
        if pixel_array is None:
            return None
        
        pixel_array = self.normalize_intensity(pixel_array)
        pixel_array = self.resize_slice(pixel_array, target_size)
        
        return pixel_array
    
    def load_study(self, study_uid: str, mode: str = 'train', 
                   target_size: int = 512, max_slices: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Load all DICOM slices for a study
        
        Args:
            study_uid: Study Instance UID
            mode: 'train' or 'test'
            target_size: Target image size (default 512)
            max_slices: Maximum slices to load (None = load all)
            
        Returns:
            Tuple of (slices array, metadata dict)
        """
        slices = []
        slice_info = []
        
        # Get all series for this study
        series_df = self.train_series_csv if mode == 'train' else self.test_series_csv
        study_series = series_df[series_df['StudyInstanceUID'] == study_uid]
        
        if len(study_series) == 0:
            print(f"⚠ No series found for study {study_uid}")
            return None, {}
        
        # Iterate through series
        for _, series_row in study_series.iterrows():
            series_uid = series_row['SeriesInstanceUID']
            series_path = self.data_root / f'{mode}_series' / study_uid / series_uid
            
            if not series_path.exists():
                print(f"⚠ Series path not found: {series_path}")
                continue
            
            # Get all DICOM files in this series
            dcm_files = sorted(series_path.glob('*.dcm'))
            
            if not dcm_files:
                print(f"⚠ No DICOM files in {series_path}")
                continue
            
            for dcm_file in dcm_files:
                pixel_array = self.process_slice(str(dcm_file), target_size)
                
                if pixel_array is not None:
                    slices.append(pixel_array)
                    slice_info.append({
                        'series_uid': series_uid,
                        'dcm_file': str(dcm_file),
                        'fluid_sensitive': series_row.get('Fluid_Sensitive', 0),
                        'fat_suppression': series_row.get('Fat_Suppression', 0),
                        'anatomical_plane': series_row.get('Anatomical_Plane', 'Unknown')
                    })
        
        if len(slices) == 0:
            print(f"⚠ No slices loaded for study {study_uid}")
            return None, {}
        
        # Convert to numpy array
        slices = np.array(slices)  # (num_slices, H, W)
        
        # Limit slices if specified
        if max_slices is not None and len(slices) > max_slices:
            indices = np.linspace(0, len(slices) - 1, max_slices, dtype=int)
            slices = slices[indices]
            slice_info = [slice_info[i] for i in indices]
        
        metadata = {
            'study_uid': study_uid,
            'num_slices': len(slices),
            'slices_shape': slices.shape,
            'num_series': len(study_series),
            'slice_info': slice_info
        }
        
        return slices, metadata
    
    def get_labels(self, study_uid: str, mode: str = 'train') -> Dict[str, float]:
        """Get all 12 abnormality labels for a study"""
        if mode == 'test':
            return {label: 0.5 for label in self.label_cols}
        
        try:
            row = self.train_csv[self.train_csv['StudyInstanceUID'] == study_uid].iloc[0]
            labels = {}
            for label in self.label_cols:
                if label in row:
                    val = row[label]
                    labels[label] = float(val) if pd.notna(val) else 0.5
                else:
                    labels[label] = 0.5
            return labels
        except IndexError:
            return {label: 0.5 for label in self.label_cols}
    
    def validate_data(self, sample_size: int = 5) -> bool:
        """Validate that data can be loaded correctly"""
        print(f"\n{'='*70}")
        print("VALIDATING DATA LOADING")
        print(f"{'='*70}")
        
        study_uids = self.train_csv['StudyInstanceUID'].head(sample_size).values
        
        for i, study_uid in enumerate(study_uids, 1):
            print(f"\n[{i}/{sample_size}] Loading study {study_uid[:20]}...")
            
            try:
                slices, metadata = self.load_study(study_uid, mode='train', target_size=512)
                
                if slices is None:
                    print("  ✗ Failed to load study")
                    return False
                
                labels = self.get_labels(study_uid)
                
                print(f"  ✓ Loaded {metadata['num_slices']} slices")
                print(f"    Shape: {metadata['slices_shape']}")
                print(f"    Series: {metadata['num_series']}")
                print(f"    Labels: {sum(1 for v in labels.values() if v != 0.5)} labeled")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                return False
        
        print(f"\n{'='*70}")
        print("✓ VALIDATION PASSED")
        print(f"{'='*70}\n")
        return True


if __name__ == "__main__":
    # Test the loader
    loader = DICOMLoader('data/')
    loader.validate_data(sample_size=3)