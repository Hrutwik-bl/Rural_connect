"""
Retrain department classifier using MobileNetV2 Transfer Learning.
This creates a separate, accurate image-based department classifier.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
import json

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 80
DEPARTMENTS = ['Electricity', 'Road', 'Water']  # Alphabetical order
DATA_DIR = 'data/images'
MODEL_PATH = 'models/dept_classifier.h5'

def load_images_and_labels():
    """Load all images (original + augmented) with department labels."""
    images = []
    labels = []
    
    dept_to_dir = {
        'Electricity': 'electricity',
        'Road': 'road', 
        'Water': 'water'
    }
    
    for dept_idx, dept_name in enumerate(DEPARTMENTS):
        dir_name = dept_to_dir[dept_name]
        dept_dir = os.path.join(DATA_DIR, dir_name)
        
        if not os.path.exists(dept_dir):
            print(f"WARNING: {dept_dir} not found!")
            continue
        
        # Load original images
        count = 0
        for fname in os.listdir(dept_dir):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dept_dir, fname)
                try:
                    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(dept_idx)
                    count += 1
                except Exception as e:
                    print(f"  Error loading {img_path}: {e}")
        
        # Load augmented images
        aug_dir = os.path.join(dept_dir, 'augmented')
        aug_count = 0
        if os.path.exists(aug_dir):
            for fname in os.listdir(aug_dir):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(aug_dir, fname)
                    try:
                        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                        img_array = np.array(img) / 255.0
                        images.append(img_array)
                        labels.append(dept_idx)
                        aug_count += 1
                    except Exception as e:
                        pass
        
        print(f"  {dept_name}: {count} original + {aug_count} augmented = {count + aug_count} total")
    
    return np.array(images), np.array(labels)

def build_model(num_classes):
    """Build MobileNetV2-based classifier."""
    # Load MobileNetV2 with ImageNet weights (no top)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def main():
    print("=" * 60)
    print("DEPARTMENT CLASSIFIER - MobileNetV2 Transfer Learning")
    print("=" * 60)
    
    # Load data
    print("\nLoading images...")
    X, y = load_images_and_labels()
    print(f"\nTotal: {len(X)} images, {len(DEPARTMENTS)} classes")
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split into train/val (80/20)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # One-hot encode
    y_train_oh = tf.keras.utils.to_categorical(y_train, len(DEPARTMENTS))
    y_val_oh = tf.keras.utils.to_categorical(y_val, len(DEPARTMENTS))
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        zoom_range=0.2,
        shear_range=0.15
    )
    
    # Build model
    print("\nBuilding MobileNetV2 model...")
    model, base_model = build_model(len(DEPARTMENTS))
    
    # Phase 1: Train classification head only (base frozen)
    print("\n--- Phase 1: Training classification head (base frozen) ---")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history1 = model.fit(
        datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val_oh),
        epochs=40,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune top layers of base model
    print("\n--- Phase 2: Fine-tuning top layers ---")
    # Unfreeze last 30 layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks2 = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    history2 = model.fit(
        datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val_oh),
        epochs=EPOCHS,
        callbacks=callbacks2,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    val_loss, val_acc = model.evaluate(X_val, y_val_oh, verbose=0)
    print(f"Validation Accuracy: {val_acc:.2%}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Per-class accuracy
    predictions = model.predict(X_val, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    print("\nPer-class results:")
    for i, dept in enumerate(DEPARTMENTS):
        mask = y_val == i
        if mask.sum() > 0:
            dept_acc = (pred_classes[mask] == i).mean()
            print(f"  {dept}: {dept_acc:.2%} ({mask.sum()} samples)")
    
    # Test on ALL original images (not augmented)
    print("\n--- Testing on ALL original images ---")
    for dept_name in DEPARTMENTS:
        dir_name = dept_name.lower() if dept_name != 'Electricity' else 'electricity'
        dept_dir = os.path.join(DATA_DIR, dir_name)
        correct = 0
        total = 0
        for fname in sorted(os.listdir(dept_dir)):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(os.path.join(dept_dir, fname)).convert('RGB').resize(IMG_SIZE)
                img_arr = np.expand_dims(np.array(img) / 255.0, axis=0)
                pred = model.predict(img_arr, verbose=0)
                pred_dept = DEPARTMENTS[np.argmax(pred)]
                is_correct = pred_dept == dept_name
                if not is_correct:
                    print(f"    WRONG: {fname} → {pred_dept} (expected {dept_name}), conf={pred[0][np.argmax(pred)]:.2f}")
                correct += int(is_correct)
                total += 1
        print(f"  {dept_name}: {correct}/{total} correct ({correct/total:.0%})")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Save class labels
    config = {
        'departments': DEPARTMENTS,
        'img_size': list(IMG_SIZE),
        'model_path': MODEL_PATH
    }
    with open('models/dept_classifier_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Config saved to models/dept_classifier_config.json")

if __name__ == '__main__':
    main()
