import os
import json
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# SETTINGS
# =========================
DATASET_DIR = "dataset"   # your folder: dataset/tumor, dataset/cyst, dataset/stone, dataset/normal
SPLIT_DIR = "outputs/sonography_final_split"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
CLASSES = ["tumor", "cyst", "stone", "normal"]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# =========================
# STANDARDIZE IMAGES
# =========================
def standardize_dataset(source_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    for category in CLASSES:
        os.makedirs(os.path.join(target_dir, category), exist_ok=True)

        src_cat = os.path.join(source_dir, category)
        dst_cat = os.path.join(target_dir, category)

        for file in os.listdir(src_cat):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                try:
                    img_path = os.path.join(src_cat, file)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMG_SIZE)

                    new_name = os.path.splitext(file)[0] + ".png"
                    img.save(os.path.join(dst_cat, new_name), "PNG")
                except Exception as e:
                    print(f"Skipped {file}: {e}")


# =========================
# SPLIT DATASET
# =========================
def split_dataset(source_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    for split in ["train", "val", "test"]:
        for category in CLASSES:
            os.makedirs(os.path.join(target_dir, split, category), exist_ok=True)

    for category in CLASSES:
        category_path = os.path.join(source_dir, category)
        images = [f for f in os.listdir(category_path) if f.endswith(".png")]

        train_imgs, temp_imgs = train_test_split(
            images, train_size=0.70, random_state=42
        )

        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=0.50, random_state=42
        )

        for img in train_imgs:
            shutil.copy2(os.path.join(category_path, img), os.path.join(target_dir, "train", category, img))

        for img in val_imgs:
            shutil.copy2(os.path.join(category_path, img), os.path.join(target_dir, "val", category, img))

        for img in test_imgs:
            shutil.copy2(os.path.join(category_path, img), os.path.join(target_dir, "test", category, img))

        print(f"{category}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")


# =========================
# MODEL
# =========================
def build_model(input_shape=(224, 224, 3), num_classes=4):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="SonoKidney_Custom_CNN")


# =========================
# MAIN TRAINING
# =========================
standardized_dir = os.path.join(OUTPUT_DIR, "dataset_standardized")

print("Standardizing images...")
standardize_dataset(DATASET_DIR, standardized_dir)

print("Splitting dataset...")
split_dataset(standardized_dir, SPLIT_DIR)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

model = build_model(num_classes=len(CLASSES))
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/best_sonokidney_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

model.load_weights("models/best_sonokidney_model.keras")

test_loss, test_acc, test_prec, test_rec = model.evaluate(test_generator)

print("\nFINAL TEST RESULTS")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")

y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("outputs/training_accuracy.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("outputs/training_loss.png")
plt.show()

model.save("models/sonokidney_final_model.keras")

with open("models/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("\nTraining complete.")
print("Saved model: models/sonokidney_final_model.keras")
print("Saved class indices: models/class_indices.json")