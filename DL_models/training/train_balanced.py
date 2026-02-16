import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Lambda, Activation, 
                                      Dropout, Flatten, Embedding, LSTM, Conv2D, 
                                      MaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from PIL import Image
import os

print("Loading training data...")

# Load dataset
df = pd.read_csv('data/text/complaints_with_valid.csv')
print(f"Total samples: {len(df)}")
print(f"Valid pairs: {(df['is_valid'] == 1).sum()}")
print(f"Invalid pairs: {(df['is_valid'] == 0).sum()}")

# PREPARE IMAGES
print("\nPreparing images...")
X_img = []
for img_path in df['image_path']:
    try:
        img = Image.open(f'data/images/{img_path}').resize((224, 224))
        img_array = np.array(img) / 255.0
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        X_img.append(img_array)
    except Exception as e:
        print(f"Warning: Error loading {img_path}: {e}")
        # Use placeholder image
        X_img.append(np.random.random((224, 224, 3)))

X_img = np.array(X_img)
print(f"Image array shape: {X_img.shape}")

# PREPARE TEXT
print("\nPreparing text...")
X_text = df['description'].values
max_words = 2000
max_len = 50

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
sequences = tokenizer.texts_to_sequences(X_text)
X_text = pad_sequences(sequences, maxlen=max_len, padding='post')
print(f"Text array shape: {X_text.shape}")

# ENCODE TARGETS
print("\nEncoding targets...")
dept_encoder = LabelEncoder()
sev_encoder = LabelEncoder()

dept_cat = keras.utils.to_categorical(dept_encoder.fit_transform(df['department']))
sev_cat = keras.utils.to_categorical(sev_encoder.fit_transform(df['severity']))
is_valid = df['is_valid'].values.astype(np.float32).reshape(-1, 1)

print(f"Department classes: {dept_encoder.classes_}")
print(f"Severity classes: {sev_encoder.classes_}")
print(f"is_valid shape: {is_valid.shape}")

# BUILD MULTIMODAL MODEL FROM SCRATCH
print("\nBuilding multimodal model...")

# Image input and CNN processing
image_input = Input(shape=(224, 224, 3), name="image_input")
x_img = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
x_img = MaxPooling2D((2, 2))(x_img)
x_img = Conv2D(64, (3, 3), activation='relu', padding='same')(x_img)
x_img = MaxPooling2D((2, 2))(x_img)
x_img = Conv2D(128, (3, 3), activation='relu', padding='same')(x_img)
x_img = GlobalAveragePooling2D()(x_img)
x_img = Dense(256, activation='relu')(x_img)
x_img = Dropout(0.3)(x_img)
img_embed = Dense(128, activation='relu', name='img_embedding')(x_img)

# Text input and RNN processing
text_input = Input(shape=(max_len,), name="text_input")
x_txt = Embedding(max_words, 64)(text_input)
x_txt = LSTM(128, return_sequences=False)(x_txt)
x_txt = Dense(256, activation='relu')(x_txt)
x_txt = Dropout(0.3)(x_txt)
txt_embed = Dense(128, activation='relu', name='txt_embedding')(x_txt)

# Normalize embeddings for cosine similarity
def normalize(x):
    return x / (tf.norm(x, axis=-1, keepdims=True) + 1e-7)

img_norm = Lambda(normalize)(img_embed)
txt_norm = Lambda(normalize)(txt_embed)

# Cosine similarity for validation
def cosine_similarity(x):
    img, txt = x
    return tf.reduce_sum(img * txt, axis=-1, keepdims=True)

similarity = Lambda(cosine_similarity)([img_norm, txt_norm])
valid_out = Dense(1, activation='sigmoid', name='is_valid')(similarity)

# Department and Severity prediction from combined embeddings
combined = Concatenate()([img_embed, txt_embed])
x = Dense(256, activation='relu')(combined)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)

dept_out = Dense(dept_cat.shape[1], activation='softmax', name='department')(x)
sev_out = Dense(sev_cat.shape[1], activation='softmax', name='severity')(x)

multimodal_model = Model(
    inputs=[image_input, text_input],
    outputs=[dept_out, sev_out, valid_out],
    name='multimodal_complaint_classifier'
)

# COMPILE
print("\nCompiling model...")

multimodal_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'department': 'categorical_crossentropy',
        'severity': 'categorical_crossentropy',
        'is_valid': 'binary_crossentropy'
    },
    loss_weights={
        'department': 1.0,
        'severity': 1.0,
        'is_valid': 3.0  # Higher weight for validation task
    },
    metrics={
        'department': 'accuracy',
        'severity': 'accuracy',
        'is_valid': 'accuracy'
    }
)

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
multimodal_model.summary()

# TRAIN
print("\n" + "="*60)
print("TRAINING MULTIMODAL MODEL")
print("="*60)
EPOCHS = 30
BATCH_SIZE = 4

# Prepare sample weights
valid_count = (is_valid == 1).sum()
invalid_count = (is_valid == 0).sum()
sample_weights = np.where(is_valid.flatten() == 1, 
                          invalid_count / len(is_valid),
                          valid_count / len(is_valid))

history = multimodal_model.fit(
    [X_img, X_text],
    {
        'department': dept_cat,
        'severity': sev_cat,
        'is_valid': is_valid
    },
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=0.2
)

# EVALUATE
print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)
results = multimodal_model.evaluate(
    [X_img, X_text],
    {
        'department': dept_cat,
        'severity': sev_cat,
        'is_valid': is_valid
    },
    verbose=0
)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Department Accuracy: {results[4]:.4f} ({results[4]*100:.2f}%)")
print(f"Severity Accuracy: {results[5]:.4f} ({results[5]*100:.2f}%)")
print(f"Validation Accuracy: {results[6]:.4f} ({results[6]*100:.2f}%)")
print("="*60)

# Save
print("\nSaving model...")
multimodal_model.save('models/multimodal_model.h5')
print("Model saved to models/multimodal_model.h5")
