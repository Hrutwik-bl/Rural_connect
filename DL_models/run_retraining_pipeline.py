"""
Complete training pipeline with data augmentation
Run this to retrain models with augmented data
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✅ Completed: {description}")
    return True


def main():
    """Execute complete retraining pipeline"""
    
    BASE_DIR = Path(__file__).parent
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  MULTIMODAL MODEL RETRAINING PIPELINE".center(58) + "║")
    print("║" + "  With Transfer Learning & Data Augmentation".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    steps = [
        (f"{sys.executable} {BASE_DIR / 'augment_text.py'}", 
         "Step 1: Text Augmentation (60 → 300+ descriptions)"),
        
        (f"{sys.executable} {BASE_DIR / 'augment_images.py'}", 
         "Step 2: Image Augmentation (60 → 300+ images)"),
        
        (f"{sys.executable} {BASE_DIR / 'train_multimodal_transfer.py'}", 
         "Step 3: Train Multimodal Model with Transfer Learning"),
    ]
    
    completed = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            completed += 1
        else:
            print(f"\n⚠ Pipeline stopped at step {completed + 1}")
            return False
    
    print(f"\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  ✅ RETRAINING COMPLETE!".center(58) + "║")
    print("║" + " "*58 + "║")
    print("║" + f"  All {completed} steps completed successfully".center(58) + "║")
    print("║" + " "*58 + "║")
    print("║" + "  📊 New Model: multimodal_model_transfer.h5".center(58) + "║")
    print("║" + "  🔑 Updated with transfer learning approach".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
