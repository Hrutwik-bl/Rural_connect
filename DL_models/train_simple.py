"""
Simplified transfer learning training - focuses on getting it working
Uses eager execution to avoid graph compilation issues
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Avoid graph compilation issues

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import json

# Configuration
IMG_SIZE = (224, 224)
MAX_LEN = 50
VOCAB_SIZE = 5000
BATCH_SIZE = 8
EPOCHS = 40  # More epochs for better learning

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def load_augmented_data():
    """Load augmented text data"""
    csv_path = DATA_DIR / "text" / "complaints_augmented.csv"
    
    if not csv_path.exists():
        print(f"⚠ Augmented CSV not found. Using original...")
        csv_path = DATA_DIR / "text" / "complaints_with_valid.csv"
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} text samples")
    return df


def load_images_with_augmentation(df):
    """Load images with augmentation support - creates both valid and invalid pairs"""
    images = []
    valid_indices = []
    validity_labels = []  # Track actual validity
    
    image_dir = DATA_DIR / "images"
    
    # First pass: collect all department images
    dept_images = {}
    for dept in ['electricity', 'road', 'water']:
        dept_dir = image_dir / dept
        if dept_dir.exists():
            aug_dir = dept_dir / "augmented"
            if aug_dir.exists():
                dept_images[dept] = list(aug_dir.glob("*.jpg"))
            if not dept_images.get(dept):
                dept_images[dept] = list(dept_dir.glob("*.jpg")) + list(dept_dir.glob("*.png"))
    
    for idx, row in df.iterrows():
        dept = row['department'].lower()
        dept_dir = image_dir / dept
        
        if not dept_dir.exists():
            continue
        
        # Get matching image (valid sample)
        image_files = dept_images.get(dept, [])
        if not image_files:
            continue
            
        try:
            # Valid sample: matching image and description
            img_path = image_files[idx % len(image_files)]
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMG_SIZE)
            images.append(np.array(img) / 255.0)
            valid_indices.append(idx)
            validity_labels.append(1.0)  # Valid pair
            
            # Create invalid sample (30% of the time): mismatched image and description
            if idx % 3 == 0:  # Every 3rd sample, add an invalid pair
                other_depts = [d for d in ['electricity', 'road', 'water'] if d != dept and dept_images.get(d)]
                if other_depts:
                    wrong_dept = other_depts[idx % len(other_depts)]
                    wrong_images = dept_images[wrong_dept]
                    if wrong_images:
                        wrong_img_path = wrong_images[idx % len(wrong_images)]
                        wrong_img = Image.open(wrong_img_path).convert('RGB')
                        wrong_img = wrong_img.resize(IMG_SIZE)
                        images.append(np.array(wrong_img) / 255.0)
                        valid_indices.append(idx)  # Same text, wrong image
                        validity_labels.append(0.0)  # Invalid pair
                        
        except Exception as e:
            pass
    
    print(f"✅ Loaded {len(images)} samples ({sum(validity_labels)} valid, {len(validity_labels) - sum(validity_labels)} invalid)")
    return np.array(images), valid_indices, np.array(validity_labels)


def build_simplified_multimodal_model(vocab_size, num_departments, num_severities):
    """Build simplified multimodal model with fine-tuning"""
    
    # CNN for images (transfer learning with MobileNetV2)
    base_cnn = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Unfreeze last 30 layers for fine-tuning on our dataset
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-30]:
        layer.trainable = False
    
    # Image processing branch
    img_input = layers.Input(shape=(*IMG_SIZE, 3), name='image')
    x = base_cnn(img_input, training=True)  # training=True for fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    img_features = layers.Dropout(0.2)(x)
    
    # Text processing branch
    text_input = layers.Input(shape=(MAX_LEN,), name='text')
    y = layers.Embedding(vocab_size, 64)(text_input)  # Increased embedding size
    y = layers.LSTM(128, dropout=0.2)(y)  # Increased LSTM size
    y = layers.Dense(128, activation='relu')(y)
    text_features = layers.Dropout(0.2)(y)
    
    # Fusion for classification
    combined = layers.Concatenate()([img_features, text_features])
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    
    # Separate validity branch - computes image-text similarity
    # This branch learns if image and text are about the same topic
    img_proj = layers.Dense(64, activation='relu')(img_features)
    txt_proj = layers.Dense(64, activation='relu')(text_features)
    # Element-wise difference captures mismatch
    diff = layers.Subtract()([img_proj, txt_proj])
    # Element-wise product captures alignment
    prod = layers.Multiply()([img_proj, txt_proj])
    validity_features = layers.Concatenate()([diff, prod, combined])
    validity_z = layers.Dense(64, activation='relu')(validity_features)
    validity_z = layers.Dropout(0.3)(validity_z)
    
    # Output heads
    dept_out = layers.Dense(num_departments, activation='softmax', name='department')(z)
    sev_out = layers.Dense(num_severities, activation='softmax', name='severity')(z)
    valid_out = layers.Dense(1, activation='sigmoid', name='validity')(validity_z)
    
    model = keras.Model(inputs=[img_input, text_input], outputs=[dept_out, sev_out, valid_out])
    return model


def train():
    """Train the multimodal model"""
    
    print("\n" + "="*60)
    print("🚀 SIMPLIFIED TRANSFER LEARNING TRAINING")
    print("="*60)
    
    # Load data
    df = load_augmented_data()
    images, valid_indices, validity_labels = load_images_with_augmentation(df)
    
    # Get descriptions/dept/severity for each sample (including duplicates for invalid pairs)
    descriptions = [df.iloc[i]['description'] for i in valid_indices]
    departments = [df.iloc[i]['department'] for i in valid_indices]
    severities = [df.iloc[i].get('severity', 'Medium') for i in valid_indices]
    
    if len(images) < 10:
        print("❌ Not enough images")
        return False
    
    # Encode labels
    dept_enc = LabelEncoder()
    sev_enc = LabelEncoder()
    dept_enc.fit(departments)
    sev_enc.fit(severities)
    
    dept_labels = dept_enc.transform(departments)
    sev_labels = sev_enc.transform(severities)
    
    print(f"\n📊 Dataset: {len(images)} samples")
    print(f"   Departments: {list(dept_enc.classes_)}")
    print(f"   Severities: {list(sev_enc.classes_)}")
    
    # Text processing
    print(f"\n📝 Processing text...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(descriptions)
    text_seqs = tokenizer.texts_to_sequences(descriptions)
    text_pad = pad_sequences(text_seqs, maxlen=MAX_LEN, padding='post')
    
    # Use validity labels from image loading (actual image-text match)
    consistency = validity_labels
    
    # Train-test split
    (img_tr, img_te, txt_tr, txt_te, d_tr, d_te, s_tr, s_te, c_tr, c_te) = train_test_split(
        images, text_pad, dept_labels, sev_labels, consistency,
        test_size=0.2, random_state=42, stratify=consistency
    )
    
    # Convert to categorical
    d_tr_cat = keras.utils.to_categorical(d_tr, num_classes=len(dept_enc.classes_))
    d_te_cat = keras.utils.to_categorical(d_te, num_classes=len(dept_enc.classes_))
    s_tr_cat = keras.utils.to_categorical(s_tr, num_classes=len(sev_enc.classes_))
    s_te_cat = keras.utils.to_categorical(s_te, num_classes=len(sev_enc.classes_))
    
    # Build and train
    print(f"\n🏗 Building model...")
    model = build_simplified_multimodal_model(VOCAB_SIZE, len(dept_enc.classes_), len(sev_enc.classes_))
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'],
        loss_weights=[1.0, 0.5, 0.8],
        metrics={'department': 'accuracy', 'severity': 'accuracy', 'validity': 'accuracy'}
    )
    
    print(f"\n🎓 Training ({EPOCHS} epochs, batch_size={BATCH_SIZE})...")
    
    # Add callbacks to save best model and prevent crashes
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "multimodal_model.h5"),
            save_best_only=True,
            monitor='val_department_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_department_accuracy',
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    model.fit(
        [img_tr, txt_tr],
        [d_tr_cat, s_tr_cat, c_tr],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([img_te, txt_te], [d_te_cat, s_te_cat, c_te]),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print(f"\n📈 Evaluating...")
    results = model.evaluate([img_te, txt_te], [d_te_cat, s_te_cat, c_te], verbose=0)
    
    print(f"\n✅ Test Results:")
    print(f"   Total Loss: {results[0]:.4f}")
    print(f"   Department Accuracy: {results[4]:.4f}")
    print(f"   Severity Accuracy: {results[5]:.4f}")
    print(f"   Validity Accuracy: {results[6]:.4f}")
    
    # Save
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "multimodal_model.h5")
    print(f"\n💾 Model saved!")
    
    return True


if __name__ == "__main__":
    success = train()
    import sys
    sys.exit(0 if success else 1)
