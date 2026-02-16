import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "data/text/complaints.csv"
MODEL_PATH = "models/rnn_text_model.h5"

VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 16

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(CSV_PATH)

texts = df["description"].astype(str).values
labels = df["department"].values

# -------------------------
# LABEL ENCODING
# -------------------------
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)
labels_cat = to_categorical(labels_encoded, num_classes)

print("Classes:", label_encoder.classes_)

# -------------------------
# TEXT TOKENIZATION
# -------------------------
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    labels_cat,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# -------------------------
# MODEL
# -------------------------
text_input = Input(shape=(MAX_LEN,))
x = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(text_input)
x = Bidirectional(LSTM(128))(x)

text_embedding = Dense(
    256,
    activation="relu",
    name="text_embedding"
)(x)

output = Dense(
    num_classes,
    activation="softmax"
)(text_embedding)

rnn_model = Model(text_input, output)

rnn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

rnn_model.summary()

# -------------------------
# TRAIN
# -------------------------
rnn_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)

# -------------------------
# EVALUATE
# -------------------------
loss, accuracy = rnn_model.evaluate(X_test, y_test)
print("RNN Test Accuracy:", accuracy)

# -------------------------
# SAVE MODEL
# -------------------------
rnn_model.save(MODEL_PATH)
print(f"✅ RNN model saved at {MODEL_PATH}")
