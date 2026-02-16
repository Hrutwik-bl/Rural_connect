import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================
# CONFIG (MATCHES YOUR ROOT)
# =====================================================
CSV_PATH = "data/text/location_verification.csv"
MODEL_PATH = "models/location_verification_model.h5"

EPOCHS = 20
BATCH_SIZE = 16

# =====================================================
# HAVERSINE DISTANCE FUNCTION
# =====================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c * 1000  # meters

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_PATH)

df["distance"] = haversine(
    df["complaint_lat"],
    df["complaint_lon"],
    df["resolved_lat"],
    df["resolved_lon"]
)

X = df[["distance"]].values
y = df["resolved"].values

print("Total samples:", len(df))
print("Resolved:", sum(y))
print("Not resolved:", len(y) - sum(y))

# =====================================================
# SCALE FEATURES
# =====================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# MODEL
# =====================================================
input_layer = Input(shape=(1,), name="distance_input")
x = Dense(16, activation="relu")(input_layer)
x = Dense(8, activation="relu")(x)
output = Dense(1, activation="sigmoid", name="resolved")(x)

verification_model = Model(input_layer, output)

verification_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

verification_model.summary()

# =====================================================
# TRAIN
# =====================================================
verification_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)

# =====================================================
# EVALUATE
# =====================================================
loss, accuracy = verification_model.evaluate(X_test, y_test)
print("📍 Location Verification Accuracy:", accuracy)

# =====================================================
# SAVE MODEL
# =====================================================
verification_model.save(MODEL_PATH)
print(f"✅ Verification model saved at {MODEL_PATH}")
