import sys
sys.path.insert(0, 'src')
from pathlib import Path
import os

data_root = Path('data/')
train_dir = data_root / 'train_series'

print("Checking first study structure...\n")

# Get first study
first_study = list(train_dir.iterdir())[0]
print(f"Study: {first_study.name}\n")

# Check what's inside
for series_dir in sorted(first_study.iterdir())[:2]:  # Check first 2 series
    print(f"Series: {series_dir.name}")
    
    # List files
    files = list(series_dir.iterdir())
    print(f"  Total items: {len(files)}")
    
    # Show file types
    for f in files[:5]:
        print(f"    - {f.name} ({f.suffix})")
    
    # Try to load one
    image_files = list(series_dir.glob('*.jpe')) + list(series_dir.glob('*.jpg'))
    print(f"  JPE/JPG files found: {len(image_files)}")
    
    if image_files:
        from PIL import Image
        try:
            img = Image.open(image_files[0])
            print(f"  ✓ Successfully loaded: {image_files[0].name}, size: {img.size}")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
    
    print()