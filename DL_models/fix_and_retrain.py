"""
Fix severity labels in training data and retrain the multimodal CNN model.
The original data had many incorrect severity labels (e.g., "Pipe burst" labeled as Low).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Lambda, Activation, 
                                      Dropout, Flatten, Embedding, LSTM, Conv2D, 
                                      MaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from PIL import Image
import os
import json

print("=" * 60)
print("STEP 1: FIXING SEVERITY LABELS")
print("=" * 60)

df = pd.read_csv('data/text/complaints_with_valid.csv')
print(f"Original data: {len(df)} samples")
print(f"Original severity distribution:\n{df['severity'].value_counts()}\n")

# Fix severity based on description content - realistic severity assignment
severity_fixes = {
    # Water complaints - fix incorrect Low labels
    "Water tank overflow creating flooding in the area": "High",
    "Pipe burst in our village, water is overflowing": "Critical",
    "There is water leakage from the pipe near my house": "Medium",
    "Sewage water is mixing with drinking water supply": "Critical",
    "Water supply is contaminated with dirt and sediment": "High",
    "Drainage system is blocked, causing water stagnation": "High",
    "Broken water tap causing wastage in the village": "Medium",
    "Water pipeline needs urgent repair": "High",
    
    # Electricity complaints - fix incorrect Low labels
    "Electrical wire creating safety hazard": "Critical",
    "High voltage wire hanging low near school": "Critical",
    "Power wire is broken and hanging near road": "Critical",
    "Electricity supply is unstable causing flickering": "Medium",
    "Electricity pole is damaged and leaning dangerously": "Critical",
    
    # Road complaints - fix where needed
    "Dangerous pothole near bus stop": "High",
    "Large potholes on main village road causing accidents": "Critical",
    "Road damage causing traffic congestion": "Medium",
    "Cracks on road after heavy rainfall": "Medium",
    "Uneven road surface near market area": "Medium",
    "Broken road near government school needs repair": "High",
    "Road needs immediate repair and patching": "High",
}

fixed_count = 0
for desc, new_sev in severity_fixes.items():
    mask = df['description'] == desc
    old_sevs = df.loc[mask, 'severity'].unique()
    if mask.any():
        changed = (df.loc[mask, 'severity'] != new_sev).sum()
        if changed > 0:
            print(f"  FIX: '{desc[:60]}...' -> {new_sev} (was {old_sevs})")
            df.loc[mask, 'severity'] = new_sev
            fixed_count += changed

print(f"\nFixed {fixed_count} labels")
print(f"New severity distribution:\n{df['severity'].value_counts()}\n")
print(f"By department:\n{df.groupby('department')['severity'].value_counts()}\n")

# Save fixed data
df.to_csv('data/text/complaints_with_valid.csv', index=False)
print("Saved fixed data to complaints_with_valid.csv")

# Also augment to balance severity classes
print("\n" + "=" * 60)
print("STEP 2: AUGMENTING DATA FOR SEVERITY BALANCE")
print("=" * 60)

# Get current counts per severity
sev_counts = df['severity'].value_counts()
target_count = sev_counts.max()
print(f"Target count per severity: {target_count}")

augmented_rows = []
for sev in ['Critical', 'High', 'Medium', 'Low']:
    subset = df[df['severity'] == sev]
    current = len(subset)
    if current == 0:
        print(f"  {sev}: 0 samples (skipping)")
        continue
    needed = target_count - current
    if needed > 0:
        # Oversample from existing rows
        extra = subset.sample(n=needed, replace=True, random_state=42)
        augmented_rows.append(extra)
        print(f"  {sev}: {current} -> {target_count} (+{needed} augmented)")
    else:
        print(f"  {sev}: {current} (no augmentation needed)")

if augmented_rows:
    df_balanced = pd.concat([df] + augmented_rows, ignore_index=True)
else:
    df_balanced = df.copy()

print(f"\nBalanced data: {len(df_balanced)} samples")
print(f"Balanced distribution:\n{df_balanced['severity'].value_counts()}")

# ==========================================
# STEP 3: RETRAIN MODEL
# ==========================================
print("\n" + "=" * 60)
print("STEP 3: PREPARING TRAINING DATA")
print("=" * 60)

# Prepare images
print("Loading images...")
X_img = []
for img_path in df_balanced['image_path']:
    try:
        img = Image.open(f'data/images/{img_path}').resize((224, 224))
        img_array = np.array(img) / 255.0
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        X_img.append(img_array)
    except Exception as e:
        print(f"  Warning: Error loading {img_path}: {e}")
        X_img.append(np.random.random((224, 224, 3)))

X_img = np.array(X_img)
print(f"Image array shape: {X_img.shape}")

# Prepare text
print("Tokenizing text...")
X_text = df_balanced['description'].values
max_words = 2000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
sequences = tokenizer.texts_to_sequences(X_text)
X_text_seq = pad_sequences(sequences, maxlen=max_len, padding='post')
print(f"Text array shape: {X_text_seq.shape}")

# Encode targets
print("Encoding targets...")
dept_encoder = LabelEncoder()
sev_encoder = LabelEncoder()

dept_cat = keras.utils.to_categorical(dept_encoder.fit_transform(df_balanced['department']))
sev_cat = keras.utils.to_categorical(sev_encoder.fit_transform(df_balanced['severity']))
is_valid = df_balanced['is_valid'].values.astype(np.float32).reshape(-1, 1)

print(f"Department classes: {list(dept_encoder.classes_)}")
print(f"Severity classes: {list(sev_encoder.classes_)}")

# Compute class weights for severity to handle residual imbalance
from sklearn.utils.class_weight import compute_class_weight
sev_labels = df_balanced['severity'].values
sev_encoded = sev_encoder.transform(sev_labels)
class_weights_sev = compute_class_weight('balanced', classes=np.unique(sev_encoded), y=sev_encoded)
sev_class_weight_dict = {i: w for i, w in enumerate(class_weights_sev)}
print(f"Severity class weights: {sev_class_weight_dict}")

# ==========================================
# STEP 4: BUILD MODEL
# ==========================================
print("\n" + "=" * 60)
print("STEP 4: BUILDING MULTIMODAL MODEL")
print("=" * 60)

# Image CNN branch
image_input = Input(shape=(224, 224, 3), name="image_input")
x_img_layer = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
x_img_layer = MaxPooling2D((2, 2))(x_img_layer)
x_img_layer = Conv2D(64, (3, 3), activation='relu', padding='same')(x_img_layer)
x_img_layer = MaxPooling2D((2, 2))(x_img_layer)
x_img_layer = Conv2D(128, (3, 3), activation='relu', padding='same')(x_img_layer)
x_img_layer = MaxPooling2D((2, 2))(x_img_layer)
x_img_layer = Conv2D(256, (3, 3), activation='relu', padding='same')(x_img_layer)
x_img_layer = GlobalAveragePooling2D()(x_img_layer)
x_img_layer = Dense(256, activation='relu')(x_img_layer)
x_img_layer = Dropout(0.4)(x_img_layer)
img_embed = Dense(128, activation='relu', name='img_embedding')(x_img_layer)

# Text LSTM branch
text_input = Input(shape=(max_len,), name="text_input")
x_txt_layer = Embedding(max_words, 64)(text_input)
x_txt_layer = LSTM(128, return_sequences=False)(x_txt_layer)
x_txt_layer = Dense(256, activation='relu')(x_txt_layer)
x_txt_layer = Dropout(0.4)(x_txt_layer)
txt_embed = Dense(128, activation='relu', name='txt_embedding')(x_txt_layer)

# Normalize for cosine similarity
def normalize(x):
    return x / (tf.norm(x, axis=-1, keepdims=True) + 1e-7)

img_norm = Lambda(normalize)(img_embed)
txt_norm = Lambda(normalize)(txt_embed)

def cosine_similarity(x):
    img_n, txt_n = x
    return tf.reduce_sum(img_n * txt_n, axis=-1, keepdims=True)

similarity = Lambda(cosine_similarity)([img_norm, txt_norm])
valid_out = Dense(1, activation='sigmoid', name='is_valid')(similarity)

# Combined features for dept + severity
combined = Concatenate()([img_embed, txt_embed])

# Department branch
dept_branch = Dense(256, activation='relu')(combined)
dept_branch = Dropout(0.3)(dept_branch)
dept_branch = Dense(128, activation='relu')(dept_branch)
dept_out = Dense(dept_cat.shape[1], activation='softmax', name='department')(dept_branch)

# Severity branch - SEPARATE from department, with MORE focus on image features
# Use image features more heavily for severity
sev_combined = Concatenate()([img_embed, img_embed, txt_embed])  # Double weight on image
sev_branch = Dense(256, activation='relu')(sev_combined)
sev_branch = Dropout(0.3)(sev_branch)
sev_branch = Dense(128, activation='relu')(sev_branch)
sev_branch = Dense(64, activation='relu')(sev_branch)
sev_out = Dense(sev_cat.shape[1], activation='softmax', name='severity')(sev_branch)

model = Model(
    inputs=[image_input, text_input],
    outputs=[dept_out, sev_out, valid_out],
    name='multimodal_complaint_classifier'
)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss={
        'department': 'categorical_crossentropy',
        'severity': 'categorical_crossentropy',
        'is_valid': 'binary_crossentropy'
    },
    loss_weights={
        'department': 1.0,
        'severity': 2.0,  # Higher weight for severity
        'is_valid': 2.0
    },
    metrics={
        'department': 'accuracy',
        'severity': 'accuracy',
        'is_valid': 'accuracy'
    }
)

model.summary()

# ==========================================
# STEP 5: TRAIN
# ==========================================
print("\n" + "=" * 60)
print("STEP 5: TRAINING MODEL")
print("=" * 60)

# Create sample weights based on severity class weights
sample_weights_array = np.array([sev_class_weight_dict[s] for s in sev_encoded])

history = model.fit(
    [X_img, X_text_seq],
    {
        'department': dept_cat,
        'severity': sev_cat,
        'is_valid': is_valid
    },
    epochs=50,
    batch_size=4,
    verbose=1,
    validation_split=0.15
)

# ==========================================
# STEP 6: EVALUATE
# ==========================================
print("\n" + "=" * 60)
print("STEP 6: EVALUATION")
print("=" * 60)

results = model.evaluate(
    [X_img, X_text_seq],
    {
        'department': dept_cat,
        'severity': sev_cat,
        'is_valid': is_valid
    },
    verbose=0
)

print(f"\nDepartment Accuracy: {results[4]*100:.2f}%")
print(f"Severity Accuracy:  {results[5]*100:.2f}%")
print(f"Validation Accuracy: {results[6]*100:.2f}%")

# Test predictions on a few samples
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
test_indices = [0, 3, 4, 8, 14, 25]
for idx in test_indices:
    if idx < len(df_balanced):
        pred = model.predict([X_img[idx:idx+1], X_text_seq[idx:idx+1]], verbose=0)
        pred_dept = dept_encoder.classes_[np.argmax(pred[0][0])]
        pred_sev = sev_encoder.classes_[np.argmax(pred[1][0])]
        actual_dept = df_balanced.iloc[idx]['department']
        actual_sev = df_balanced.iloc[idx]['severity']
        desc = df_balanced.iloc[idx]['description'][:60]
        print(f"  [{idx}] '{desc}...'")
        print(f"       Dept: {pred_dept} (actual: {actual_dept}), Sev: {pred_sev} (actual: {actual_sev})")
        print(f"       Sev probs: {dict(zip(sev_encoder.classes_, [f'{p:.3f}' for p in pred[1][0]]))}")

# ==========================================
# STEP 7: SAVE
# ==========================================
print("\n" + "=" * 60)
print("STEP 7: SAVING MODEL AND ENCODERS")
print("=" * 60)

model.save('models/multimodal_model.h5')
print("Model saved to models/multimodal_model.h5")

# Save tokenizer
tokenizer_data = {
    "word_index": tokenizer.word_index,
    "max_len": max_len
}
with open('models/tokenizer_transfer.json', 'w') as f:
    json.dump(tokenizer_data, f)
print("Tokenizer saved")

# Save encoders
with open('models/dept_encoder_transfer.json', 'w') as f:
    json.dump({"classes": list(dept_encoder.classes_)}, f)
print(f"Department encoder saved: {list(dept_encoder.classes_)}")

with open('models/sev_encoder_transfer.json', 'w') as f:
    json.dump({"classes": list(sev_encoder.classes_)}, f)
print(f"Severity encoder saved: {list(sev_encoder.classes_)}")

print("\n" + "=" * 60)
print("RETRAINING COMPLETE!")
print("=" * 60)
