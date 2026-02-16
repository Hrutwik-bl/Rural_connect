import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 3   # electricity, road, water

DATA_DIR = "data/images"
MODEL_PATH = "models/cnn_image_model.h5"

# -------------------------
# DATA GENERATOR
# -------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,   # 80% train, 20% validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("Class labels:", train_data.class_indices)

# -------------------------
# MODEL
# -------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

image_input = Input(shape=(224, 224, 3))
x = base_model(image_input)
x = GlobalAveragePooling2D()(x)

embedding = Dense(256, activation="relu", name="image_embedding")(x)
output = Dense(NUM_CLASSES, activation="softmax")(embedding)

cnn_model = Model(image_input, output)

cnn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_model.summary()

# -------------------------
# TRAIN
# -------------------------
cnn_model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# -------------------------
# EVALUATE
# -------------------------
loss, accuracy = cnn_model.evaluate(val_data)
print("CNN Validation Accuracy:", accuracy)

# -------------------------
# SAVE MODEL
# -------------------------
cnn_model.save(MODEL_PATH)
print(f"✅ CNN model saved at {MODEL_PATH}")
