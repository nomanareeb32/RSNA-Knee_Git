"""
Inference Script for RSNA Knee Abnormality Detection
Generates predictions and submission CSV for Kaggle
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys

sys.path.insert(0, 'src')

from image_loader import ImageLoader
from models import KneeAbnormalityModel


def predict_study(study_uid: str, loader: ImageLoader, model, device, 
                  max_images: int = 32) -> np.ndarray:
    """
    Predict for a single study
    
    Args:
        study_uid: Study Instance UID
        loader: ImageLoader instance
        model: Trained model
        device: torch.device
        max_images: Maximum images per study
        
    Returns:
        Predictions array (12,)
    """
    # Load study images (3-channel RGB)
    images, _ = loader.load_study(study_uid, mode='test', max_images=None)
    
    if images is None:
        # Return neutral predictions
        return np.array([0.5] * 12)
    
    # Pad or truncate to max_images
    # images shape: (N, H, W, 3)
    if len(images) < max_images:
        pad_size = max_images - len(images)
        images = np.pad(images, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
    else:
        indices = np.linspace(0, len(images) - 1, max_images, dtype=int)
        images = images[indices]
    
    # Convert to tensor: (max_images, H, W, 3) -> (1, max_images, H, W, 3)
    images_tensor = torch.from_numpy(images).unsqueeze(0).float()
    
    # Predict
    model.eval()
    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        output = model(images_tensor)
    
    predictions = output[0].cpu().numpy()  # (12,)
    
    return predictions


def generate_submission(model_path: str, output_path: str = 'submissions/submission.csv'):
    """
    Generate submission CSV for competition
    
    Args:
        model_path: Path to trained model checkpoint
        output_path: Path to save submission CSV
    """
    print("\n" + "="*70)
    print("RSNA KNEE ABNORMALITY DETECTION - INFERENCE")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    model = KneeAbnormalityModel(
        backbone='resnet50',
        pretrained=True,
        aggregation='max',
        num_abnormalities=12
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Loaded model from {model_path}\n")
    
    # Load data
    print("Loading test data metadata...")
    test_df = pd.read_csv('data/test.csv')
    loader = ImageLoader('data/')
    
    label_cols = [
        'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
        'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
        'Synovitis', "Baker's", 'Contusion', 'Fracture'
    ]
    
    print(f"Found {len(test_df)} test studies\n")
    
    # Generate predictions
    print(f"Generating predictions...\n")
    results = []
    start_time = time.time()
    
    for study_uid in tqdm(test_df['StudyInstanceUID'], desc='Predicting'):
        predictions = predict_study(study_uid, loader, model, device, max_images=32)
        results.append([study_uid] + predictions.tolist())
    
    elapsed = time.time() - start_time
    
    # Create submission
    submission_df = pd.DataFrame(
        results,
        columns=['StudyInstanceUID'] + label_cols
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n" + "="*70)
    print("INFERENCE COMPLETE!")
    print("="*70)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Time per study: {elapsed/len(test_df):.2f} seconds")
    print(f"Submission saved: {output_path}")
    print("="*70 + "\n")
    
    # Show sample
    print("Sample submission:")
    print(submission_df.head())
    
    # Verify format
    print(f"\nVerification:")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Columns: {submission_df.columns.tolist()}")
    print(f"  No NaN: {submission_df.isna().sum().sum() == 0}")
    print(f"  Values in [0,1]: {((submission_df.iloc[:, 1:] >= 0) & (submission_df.iloc[:, 1:] <= 1)).all().all()}")


if __name__ == "__main__":
    # Find best model
    checkpoint_dir = Path('checkpoints')
    best_models = list(checkpoint_dir.glob('best_model_auc_*.pt'))
    
    if not best_models:
        print("❌ No model checkpoints found!")
        print("   Train a model first using: python train_jpeg.py")
        exit(1)
    
    # Use most recent best model
    model_path = max(best_models, key=lambda p: p.stat().st_mtime)
    
    print(f"Using model: {model_path.name}")
    generate_submission(str(model_path))