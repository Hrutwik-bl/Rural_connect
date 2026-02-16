"""
Complete Multimodal Training Script
- Uses augmented datasets (images and text)
- Trains for: Department, Severity, Validity (image-text match)
- Creates both valid pairs (matching) and invalid pairs (mismatched)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import json
import random

# Configuration
IMG_SIZE = (224, 224)
MAX_LEN = 50
VOCAB_SIZE = 5000
BATCH_SIZE = 16
EPOCHS = 30

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


def load_all_images_by_department():
    """Load ALL images (original + augmented) organized by department"""
    image_dir = DATA_DIR / "images"
    dept_images = {}
    
    for dept in ['electricity', 'road', 'water']:
        dept_dir = image_dir / dept
        if not dept_dir.exists():
            continue
            
        images = []
        
        # Load original images
        for img_path in list(dept_dir.glob("*.jpg")) + list(dept_dir.glob("*.png")):
            try:
                img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                images.append(np.array(img) / 255.0)
            except:
                pass
        
        # Load augmented images
        aug_dir = dept_dir / "augmented"
        if aug_dir.exists():
            for img_path in aug_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                    images.append(np.array(img) / 255.0)
                except:
                    pass
        
        dept_images[dept] = np.array(images) if images else np.array([])
        print(f"  {dept.capitalize()}: {len(images)} images")
    
    return dept_images


def create_training_data():
    """Create training data with valid and invalid pairs"""
    
    print("\n📊 Loading data...")
    
    # Load text data
    csv_path = DATA_DIR / "text" / "complaints_augmented.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "text" / "complaints.csv"
    
    df = pd.read_csv(csv_path)
    print(f"  Text samples: {len(df)}")
    
    # Load images by department
    dept_images = load_all_images_by_department()
    
    departments = ['Electricity', 'Road', 'Water']
    severities = ['Low', 'Medium', 'High', 'Critical']
    
    # Create dataset
    all_images = []
    all_texts = []
    all_depts = []
    all_sevs = []
    all_validity = []
    
    print("\n🔄 Creating training pairs...")
    
    for idx, row in df.iterrows():
        dept = row['department']
        desc = row['description']
        sev = row.get('severity', 'Medium')
        
        dept_lower = dept.lower()
        if dept_lower not in dept_images or len(dept_images[dept_lower]) == 0:
            continue
        
        dept_imgs = dept_images[dept_lower]
        
        # Create VALID pair - matching image and text
        img_idx = idx % len(dept_imgs)
        all_images.append(dept_imgs[img_idx])
        all_texts.append(desc)
        all_depts.append(dept)
        all_sevs.append(sev)
        all_validity.append(1.0)  # VALID - image matches text
        
        # Create INVALID pair (50% of samples) - mismatched image and text
        if idx % 2 == 0:
            other_depts = [d for d in ['electricity', 'road', 'water'] if d != dept_lower]
            wrong_dept = random.choice(other_depts)
            if len(dept_images[wrong_dept]) > 0:
                wrong_img = dept_images[wrong_dept][idx % len(dept_images[wrong_dept])]
                all_images.append(wrong_img)
                all_texts.append(desc)  # Same text, wrong image
                all_depts.append(dept)  # Label based on text
                all_sevs.append(sev)
                all_validity.append(0.0)  # INVALID - image doesn't match text
    
    print(f"  Total samples: {len(all_images)}")
    print(f"  Valid pairs: {sum(all_validity):.0f}")
    print(f"  Invalid pairs: {len(all_validity) - sum(all_validity):.0f}")
    
    return (np.array(all_images), all_texts, all_depts, all_sevs, np.array(all_validity))


def build_multimodal_model(vocab_size, num_departments, num_severities):
    """Build multimodal model for department, severity, and validity prediction"""
    
    # Image branch - MobileNetV2 with fine-tuning
    base_cnn = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Fine-tune last 20 layers
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-20]:
        layer.trainable = False
    
    img_input = layers.Input(shape=(*IMG_SIZE, 3), name='image')
    x = base_cnn(img_input, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    img_features = layers.Dense(128, activation='relu', name='img_embed')(x)
    
    # Text branch - LSTM
    text_input = layers.Input(shape=(MAX_LEN,), name='text')
    y = layers.Embedding(vocab_size, 64)(text_input)
    y = layers.LSTM(128, dropout=0.2, return_sequences=True)(y)
    y = layers.LSTM(64, dropout=0.2)(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    text_features = layers.Dense(128, activation='relu', name='text_embed')(y)
    
    # Combined features for classification
    combined = layers.Concatenate()([img_features, text_features])
    z = layers.Dense(256, activation='relu')(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(128, activation='relu')(z)
    
    # Validity branch - learns image-text relationship
    # Project both to same space and compare
    img_proj = layers.Dense(64, activation='tanh', name='img_proj')(img_features)
    txt_proj = layers.Dense(64, activation='tanh', name='txt_proj')(text_features)
    
    # Multiple comparison methods
    diff = layers.Subtract()([img_proj, txt_proj])
    diff_sq = layers.Multiply()([diff, diff])  # Squared difference
    prod = layers.Multiply()([img_proj, txt_proj])  # Element-wise product
    
    validity_combined = layers.Concatenate()([diff_sq, prod, img_proj, txt_proj])
    validity_z = layers.Dense(64, activation='relu')(validity_combined)
    validity_z = layers.BatchNormalization()(validity_z)
    validity_z = layers.Dropout(0.3)(validity_z)
    validity_z = layers.Dense(32, activation='relu')(validity_z)
    
    # Output heads
    dept_out = layers.Dense(num_departments, activation='softmax', name='department')(z)
    sev_out = layers.Dense(num_severities, activation='softmax', name='severity')(z)
    valid_out = layers.Dense(1, activation='sigmoid', name='validity')(validity_z)
    
    model = keras.Model(
        inputs=[img_input, text_input],
        outputs=[dept_out, sev_out, valid_out]
    )
    
    return model


def train():
    """Main training function"""
    
    print("\n" + "="*60)
    print("🚀 COMPLETE MULTIMODAL TRAINING")
    print("="*60)
    
    # Create training data
    images, texts, depts, sevs, validity = create_training_data()
    
    if len(images) < 20:
        print("❌ Not enough data")
        return False
    
    # Encode labels
    dept_enc = LabelEncoder()
    sev_enc = LabelEncoder()
    dept_enc.fit(['Electricity', 'Road', 'Water'])
    sev_enc.fit(['Low', 'Medium', 'High', 'Critical'])
    
    dept_labels = dept_enc.transform(depts)
    sev_labels = sev_enc.transform(sevs)
    
    print(f"\n📊 Dataset ready:")
    print(f"   Samples: {len(images)}")
    print(f"   Departments: {list(dept_enc.classes_)}")
    print(f"   Severities: {list(sev_enc.classes_)}")
    
    # Text processing
    print(f"\n📝 Processing text...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    text_seqs = tokenizer.texts_to_sequences(texts)
    text_pad = pad_sequences(text_seqs, maxlen=MAX_LEN, padding='post')
    
    # Train-test split (stratified by validity)
    (img_tr, img_te, txt_tr, txt_te, 
     d_tr, d_te, s_tr, s_te, v_tr, v_te) = train_test_split(
        images, text_pad, dept_labels, sev_labels, validity,
        test_size=0.2, random_state=42, stratify=validity
    )
    
    # Convert to categorical
    d_tr_cat = keras.utils.to_categorical(d_tr, num_classes=len(dept_enc.classes_))
    d_te_cat = keras.utils.to_categorical(d_te, num_classes=len(dept_enc.classes_))
    s_tr_cat = keras.utils.to_categorical(s_tr, num_classes=len(sev_enc.classes_))
    s_te_cat = keras.utils.to_categorical(s_te, num_classes=len(sev_enc.classes_))
    
    print(f"   Train: {len(img_tr)} samples")
    print(f"   Test: {len(img_te)} samples")
    print(f"   Train valid/invalid: {v_tr.sum():.0f}/{len(v_tr)-v_tr.sum():.0f}")
    
    # Build model
    print(f"\n🏗 Building model...")
    model = build_multimodal_model(VOCAB_SIZE, len(dept_enc.classes_), len(sev_enc.classes_))
    
    # Compile with balanced losses
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss={
            'department': 'categorical_crossentropy',
            'severity': 'categorical_crossentropy',
            'validity': 'binary_crossentropy'
        },
        loss_weights={
            'department': 1.0,
            'severity': 0.5,
            'validity': 1.5  # Higher weight for validity
        },
        metrics={
            'department': 'accuracy',
            'severity': 'accuracy',
            'validity': 'accuracy'
        }
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "multimodal_model.h5"),
            save_best_only=True,
            monitor='val_validity_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_validity_accuracy',
            patience=8,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    print(f"\n🎓 Training ({EPOCHS} epochs)...")
    history = model.fit(
        [img_tr, txt_tr],
        {'department': d_tr_cat, 'severity': s_tr_cat, 'validity': v_tr},
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(
            [img_te, txt_te],
            {'department': d_te_cat, 'severity': s_te_cat, 'validity': v_te}
        ),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print(f"\n📈 Final Evaluation...")
    results = model.evaluate(
        [img_te, txt_te],
        {'department': d_te_cat, 'severity': s_te_cat, 'validity': v_te},
        verbose=0
    )
    
    # Get metrics
    metrics = dict(zip(model.metrics_names, results))
    
    print(f"\n✅ Results:")
    print(f"   Department Accuracy: {metrics.get('department_accuracy', 0):.2%}")
    print(f"   Severity Accuracy: {metrics.get('severity_accuracy', 0):.2%}")
    print(f"   Validity Accuracy: {metrics.get('validity_accuracy', 0):.2%}")
    
    # Save model and encoders
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "multimodal_model.h5")
    
    # Save tokenizer
    with open(MODELS_DIR / "tokenizer_transfer.json", "w") as f:
        json.dump({"word_index": tokenizer.word_index}, f)
    
    # Save encoders
    with open(MODELS_DIR / "dept_encoder_transfer.json", "w") as f:
        json.dump({"classes": list(dept_enc.classes_)}, f)
    
    with open(MODELS_DIR / "sev_encoder_transfer.json", "w") as f:
        json.dump({"classes": list(sev_enc.classes_)}, f)
    
    print(f"\n💾 Model and encoders saved!")
    
    return True


if __name__ == "__main__":
    success = train()
    import sys
    sys.exit(0 if success else 1)
