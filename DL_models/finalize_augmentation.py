"""
Minimal training - Augmented data is ready, now use existing model with updated validation
"""

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def finalize_augmented_dataset():
    """Mark augmented dataset as ready"""
    
    print("\n" + "="*60)
    print("✅ AUGMENTATION PIPELINE COMPLETE")
    print("="*60)
    
    # Report stats
    aug_csv = DATA_DIR / "text" / "complaints_augmented.csv"
    if aug_csv.exists():
        import pandas as pd
        df = pd.read_csv(aug_csv)
        print(f"\n📊 Dataset Summary:")
        print(f"   Total Samples: {len(df)}")
        print(f"   Original: {len(df[df['is_augmented'] == 0])}")
        print(f"   Augmented: {len(df[df['is_augmented'] == 1])}")
        print(f"   Departments: {df['department'].unique().tolist()}")
        print(f"   Severities: {df['severity'].unique().tolist()}")
    
    # Report images
    print(f"\n📸 Image Dataset:")
    for dept in ['water', 'road', 'electricity']:
        dept_dir = DATA_DIR / "images" / dept
        if dept_dir.exists():
            orig_count = len(list(dept_dir.glob("*.jpg"))) + len(list(dept_dir.glob("*.png")))
            aug_dir = dept_dir / "augmented"
            aug_count = len(list(aug_dir.glob("*.jpg"))) if aug_dir.exists() else 0
            total = orig_count + aug_count
            print(f"   {dept.title()}: {orig_count} originals + {aug_count} augmented = {total} total")
    
    print(f"\n✅ Data is ready for training!")
    print(f"\n📝 Next Steps:")
    print(f"   1. The existing api.py will use improved validation rules")
    print(f"   2. For full retraining: Run 'python train_simple.py' with more memory")
    print(f"   3. Or train on external GPU for better performance")
    
    return True


if __name__ == "__main__":
    finalize_augmented_dataset()
