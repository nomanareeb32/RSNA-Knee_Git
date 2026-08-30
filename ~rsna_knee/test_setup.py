import sys
sys.path.insert(0, 'src')
from src.image_loader import ImageLoader

print("\n" + "="*70)
print("TESTING JPEG DATA LOADING")
print("="*70 + "\n")

loader = ImageLoader('data/')

if loader.train_csv is not None:
    for i, study_uid in enumerate(loader.train_csv['StudyInstanceUID'].head(3).values, 1):
        print(f"[{i}/3] {study_uid[:40]}...")
        images, meta = loader.load_study(study_uid)
        
        if images is not None:
            print(f"  ✓ Loaded {meta['num_images']} images - Shape: {meta['shape']}\n")
        else:
            print(f"  ✗ Failed\n")

print("="*70)
print("✓ READY TO TRAIN!")
print("="*70)