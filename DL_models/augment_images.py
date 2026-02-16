"""
Image augmentation for limited complaint dataset
Generates variations of complaint images using rotation, flip, zoom, etc.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import random

def augment_image(image_path, output_dir, num_augmentations=4):
    """Generate augmented versions of a single image"""
    
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Could not load {image_path}: {e}")
        return 0
    
    augmented_count = 0
    filename = Path(image_path).stem
    
    # Save original
    original_output = output_dir / f"{filename}_aug_0.jpg"
    img.save(original_output, quality=95)
    augmented_count += 1
    
    for aug_idx in range(1, num_augmentations + 1):
        try:
            aug_img = img.copy()
            
            # Random augmentation pipeline
            augmentation = random.randint(0, 5)
            
            if augmentation == 0:  # Rotation
                angle = random.randint(-15, 15)
                aug_img = aug_img.rotate(angle, expand=False, fillcolor='white')
            
            elif augmentation == 1:  # Horizontal flip
                aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            elif augmentation == 2:  # Brightness adjustment
                enhancer = ImageEnhance.Brightness(aug_img)
                factor = random.uniform(0.8, 1.2)
                aug_img = enhancer.enhance(factor)
            
            elif augmentation == 3:  # Contrast adjustment
                enhancer = ImageEnhance.Contrast(aug_img)
                factor = random.uniform(0.8, 1.2)
                aug_img = enhancer.enhance(factor)
            
            elif augmentation == 4:  # Zoom (crop and resize)
                w, h = aug_img.size
                crop_factor = random.uniform(0.85, 0.95)
                crop_w, crop_h = int(w * crop_factor), int(h * crop_factor)
                left = random.randint(0, w - crop_w)
                top = random.randint(0, h - crop_h)
                aug_img = aug_img.crop((left, top, left + crop_w, top + crop_h))
                aug_img = aug_img.resize((w, h), Image.Resampling.LANCZOS)
            
            elif augmentation == 5:  # Saturation
                enhancer = ImageEnhance.Color(aug_img)
                factor = random.uniform(0.8, 1.2)
                aug_img = enhancer.enhance(factor)
            
            # Save augmented image
            output_path = output_dir / f"{filename}_aug_{aug_idx}.jpg"
            aug_img.save(output_path, quality=95)
            augmented_count += 1
            
        except Exception as e:
            print(f"⚠ Augmentation {aug_idx} failed for {filename}: {e}")
    
    return augmented_count


def augment_dataset(base_data_dir, augmentations_per_image=4):
    """Augment all images in dataset folders"""
    
    base_path = Path(base_data_dir)
    image_dir = base_path / "images"
    
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    departments = ['water', 'road', 'electricity']
    total_augmented = 0
    
    for dept in departments:
        dept_dir = image_dir / dept
        if not dept_dir.exists():
            print(f"⚠ Department directory not found: {dept_dir}")
            continue
        
        # Create augmented subdirectory
        aug_dir = dept_dir / "augmented"
        aug_dir.mkdir(exist_ok=True)
        
        # Augment all images in this department
        image_files = list(dept_dir.glob("*.jpg")) + list(dept_dir.glob("*.png"))
        
        if not image_files:
            print(f"⚠ No images found in {dept_dir}")
            continue
        
        print(f"\n📸 Augmenting {dept} ({len(image_files)} images)...")
        
        for image_path in image_files:
            count = augment_image(image_path, aug_dir, num_augmentations=augmentations_per_image)
            total_augmented += count
        
        print(f"✅ {dept}: {len(image_files) * (augmentations_per_image + 1)} images total")
    
    print(f"\n🎉 Total augmented images created: {total_augmented}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    data_dir = BASE_DIR / "data"
    
    augment_dataset(data_dir, augmentations_per_image=4)
