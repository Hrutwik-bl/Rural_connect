"""
Transfer learning training for multimodal model with augmented data
Uses pretrained CNN (MobileNetV2) + RNN on augmented dataset
Includes image-text consistency labels
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
VOCAB_SIZE = 10000
BATCH_SIZE = 16
EPOCHS = 50

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
    """Load images with augmentation support"""
    images = []
    valid_indices = []
    
    image_dir = DATA_DIR / "images"
    
    for idx, row in df.iterrows():
        dept = row['department'].lower()
        dept_dir = image_dir / dept
        
        if not dept_dir.exists():
            continue
        
        # Try augmented first, then original
        image_files = list((dept_dir / "augmented").glob("*.jpg")) if (dept_dir / "augmented").exists() else []
        if not image_files:
            image_files = list(dept_dir.glob("*.jpg")) + list(dept_dir.glob("*.png"))
        
        if image_files:
            try:
                img_path = image_files[idx % len(image_files)]  # Cycle through available images
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                images.append(np.array(img) / 255.0)
                valid_indices.append(idx)
            except Exception as e:
                print(f"⚠ Could not load image for {dept}: {e}")
    
    print(f"✅ Loaded {len(images)} images")
    return np.array(images), valid_indices


def build_transfer_learning_cnn():
    """Build CNN using transfer learning (MobileNetV2)"""
    
    # Load pretrained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base layers (feature extraction mode)
    base_model.trainable = False
    
    # Add custom top layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
    ], name="cnn_feature_extractor")
    
    return model


def build_rnn_text_model(vocab_size, max_len):
    """Build RNN for text processing"""
    model = keras.Sequential([
        layers.Embedding(vocab_size, 64, input_length=max_len),
        layers.LSTM(128, return_sequences=True, dropout=0.3),
        layers.LSTM(64, dropout=0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
    ], name="rnn_text_processor")
    
    return model


def build_multimodal_model(dept_encoder, sev_encoder, vocab_size):
    """Build complete multimodal model with transfer learning"""
    
    num_departments = len(dept_encoder.classes_)
    num_severities = len(sev_encoder.classes_)
    
    # Image input and processing
    img_input = layers.Input(shape=(*IMG_SIZE, 3), name='image_input')
    cnn_model = build_transfer_learning_cnn()
    img_features = cnn_model(img_input)
    
    # Text input and processing
    text_input = layers.Input(shape=(MAX_LEN,), name='text_input')
    rnn_model = build_rnn_text_model(vocab_size, MAX_LEN)
    text_features = rnn_model(text_input)
    
    # Normalize features
    def normalize(x):
        return x / (tf.norm(x, axis=-1, keepdims=True) + 1e-7)
    
    img_normalized = layers.Lambda(normalize)(img_features)
    text_normalized = layers.Lambda(normalize)(text_features)
    
    # Fusion layer
    def cosine_similarity(x):
        img, txt = x
        return tf.reduce_sum(img * txt, axis=-1, keepdims=True)
    
    similarity = layers.Lambda(cosine_similarity, name='similarity')([img_normalized, text_normalized])
    
    # Concatenate for final predictions
    combined = layers.Concatenate()([img_features, text_features, similarity])
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.2)(combined)
    
    # Output heads
    dept_output = layers.Dense(num_departments, activation='softmax', name='department')(combined)
    sev_output = layers.Dense(num_severities, activation='softmax', name='severity')(combined)
    valid_output = layers.Dense(1, activation='sigmoid', name='validity')(combined)
    
    # Build model
    model = keras.Model(
        inputs=[img_input, text_input],
        outputs=[dept_output, sev_output, valid_output]
    )
    
    return model, cnn_model, rnn_model


def create_consistency_labels(descriptions, departments):
    """Create binary consistency labels based on keyword matching"""
    
    dept_keywords = {
        'Water': ['water', 'pipe', 'leak', 'supply', 'drainage', 'tap', 'sewer', 'sewage'],
        'Road': ['road', 'pothole', 'street', 'path', 'highway', 'pavement', 'crack'],
        'Electricity': ['electric', 'electricity', 'power', 'light', 'wire', 'pole', 'current'],
    }
    
    consistency = []
    for desc, dept in zip(descriptions, departments):
        text_lower = desc.lower()
        keywords = dept_keywords.get(dept, [])
        is_consistent = 1.0 if any(kw in text_lower for kw in keywords) else 0.0
        consistency.append(is_consistent)
    
    return np.array(consistency)


def train_multimodal_model():
    """Train the multimodal model with transfer learning"""
    
    print("=" * 60)
    print("🚀 TRANSFER LEARNING MULTIMODAL MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_augmented_data()
    images, valid_indices = load_images_with_augmentation(df)
    df = df.iloc[valid_indices].reset_index(drop=True)
    
    if len(images) < 10:
        print("❌ Not enough images loaded. Please check data directory.")
        return
    
    descriptions = df['description'].values
    departments = df['department'].values
    severities = df.get('severity', pd.Series(['Medium'] * len(df))).values
    
    # Encode labels
    dept_encoder = LabelEncoder()
    sev_encoder = LabelEncoder()
    
    dept_encoded = dept_encoder.fit_transform(departments)
    sev_encoded = sev_encoder.fit_transform(severities)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Images: {len(images)}")
    print(f"   Texts: {len(descriptions)}")
    print(f"   Departments: {dept_encoder.classes_}")
    print(f"   Severities: {sev_encoder.classes_}")
    
    # Text processing
    print(f"\n📝 Processing text...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(descriptions)
    
    text_sequences = tokenizer.texts_to_sequences(descriptions)
    text_padded = pad_sequences(text_sequences, maxlen=MAX_LEN, padding='post')
    
    # Create consistency labels
    consistency_labels = create_consistency_labels(descriptions, departments)
    
    # Train-test split
    (img_train, img_test, text_train, text_test, 
     dept_train, dept_test, sev_train, sev_test,
     cons_train, cons_test) = train_test_split(
        images, text_padded, dept_encoded, sev_encoded, consistency_labels,
        test_size=0.2, random_state=42
    )
    
    dept_train_cat = tf.keras.utils.to_categorical(dept_train, num_classes=len(dept_encoder.classes_))
    dept_test_cat = tf.keras.utils.to_categorical(dept_test, num_classes=len(dept_encoder.classes_))
    
    sev_train_cat = tf.keras.utils.to_categorical(sev_train, num_classes=len(sev_encoder.classes_))
    sev_test_cat = tf.keras.utils.to_categorical(sev_test, num_classes=len(sev_encoder.classes_))
    
    # Build model
    print(f"\n🏗 Building transfer learning model...")
    model, cnn_model, rnn_model = build_multimodal_model(dept_encoder, sev_encoder, VOCAB_SIZE)
    
    # Compile with weighted loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'department': 'categorical_crossentropy',
            'severity': 'categorical_crossentropy',
            'validity': 'binary_crossentropy'
        },
        loss_weights={
            'department': 1.0,
            'severity': 0.5,
            'validity': 1.0
        },
        metrics={
            'department': 'accuracy',
            'severity': 'accuracy',
            'validity': 'accuracy'
        }
    )
    
    # Train
    print(f"\n🎓 Training model...")
    history = model.fit(
        [img_train, text_train],
        [dept_train_cat, sev_train_cat, cons_train],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([img_test, text_test], [dept_test_cat, sev_test_cat, cons_test]),
        verbose=1
    )
    
    # Evaluate
    print(f"\n📈 Evaluating on test set...")
    test_loss, *test_metrics = model.evaluate([img_test, text_test], [dept_test_cat, sev_test_cat, cons_test])
    
    print(f"\n✅ Test Results:")
    print(f"   Department Accuracy: {test_metrics[0]:.4f}")
    print(f"   Severity Accuracy: {test_metrics[2]:.4f}")
    print(f"   Validity Accuracy: {test_metrics[4]:.4f}")
    
    # Save model and encoders
    MODELS_DIR.mkdir(exist_ok=True)
    
    model.save(MODELS_DIR / "multimodal_model_transfer.h5")
    print(f"\n💾 Model saved to: {MODELS_DIR / 'multimodal_model_transfer.h5'}")
    
    # Save encoders
    with open(MODELS_DIR / "dept_encoder_transfer.json", "w") as f:
        json.dump({"classes": dept_encoder.classes_.tolist()}, f)
    
    with open(MODELS_DIR / "sev_encoder_transfer.json", "w") as f:
        json.dump({"classes": sev_encoder.classes_.tolist()}, f)
    
    # Save tokenizer config
    with open(MODELS_DIR / "tokenizer_transfer.json", "w") as f:
        json.dump({
            "word_index": tokenizer.word_index,
            "vocab_size": VOCAB_SIZE,
            "max_len": MAX_LEN
        }, f)
    
    print(f"✅ Encoders and tokenizer saved")
    
    return model, history


if __name__ == "__main__":
    model, history = train_multimodal_model()
