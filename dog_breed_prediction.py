"""
Dog Breed Prediction System
============================
Uses Transfer Learning with MobileNetV2 (TensorFlow/Keras) + OpenCV for image preprocessing.
Trains on Stanford Dogs Dataset structure (or any organized folder dataset).

Author: Nishita Thakur
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE      = 224          # MobileNetV2 default input
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.001
DATASET_DIR   = "dataset"    # Folder structure: dataset/train/breed_name/*.jpg
MODEL_PATH    = "dog_breed_model.h5"


# ─────────────────────────────────────────────
# 1. DATA AUGMENTATION & GENERATORS
# ─────────────────────────────────────────────
def build_generators(dataset_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_generator, val_generator


# ─────────────────────────────────────────────
# 2. MODEL ARCHITECTURE (Transfer Learning)
# ─────────────────────────────────────────────
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze base layers initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────
def train_model(dataset_dir=DATASET_DIR):
    print("\n[INFO] Loading dataset...")
    train_gen, val_gen = build_generators(dataset_dir)
    num_classes = len(train_gen.class_indices)
    print(f"[INFO] Found {num_classes} breeds | Train: {train_gen.samples} | Val: {val_gen.samples}")

    print("\n[INFO] Building model...")
    model, base_model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    ]

    print("\n[INFO] Phase 1: Training top layers...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks
    )

    # Fine-tuning: Unfreeze last 30 layers
    print("\n[INFO] Phase 2: Fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"\n[INFO] Model saved to {MODEL_PATH}")
    plot_training(history1, history2)

    # Save class labels
    class_indices = train_gen.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    np.save("class_labels.npy", idx_to_class)
    print("[INFO] Class labels saved to class_labels.npy")

    return model, idx_to_class


# ─────────────────────────────────────────────
# 4. TRAINING PLOTS
# ─────────────────────────────────────────────
def plot_training(history1, history2):
    acc  = history1.history["accuracy"]  + history2.history["accuracy"]
    val  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    vloss= history1.history["val_loss"] + history2.history["val_loss"]

    epochs_range = range(len(acc))
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc,  label="Train Accuracy",  color="#2196F3")
    plt.plot(epochs_range, val,  label="Val Accuracy",    color="#4CAF50")
    plt.axvline(x=9, color="gray", linestyle="--", label="Fine-tune start")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss,  label="Train Loss", color="#F44336")
    plt.plot(epochs_range, vloss, label="Val Loss",   color="#FF9800")
    plt.axvline(x=9, color="gray", linestyle="--", label="Fine-tune start")
    plt.title("Model Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("[INFO] Training plot saved.")


# ─────────────────────────────────────────────
# 5. IMAGE PREPROCESSING WITH OPENCV
# ─────────────────────────────────────────────
def preprocess_image(image_path):
    """Load and preprocess an image using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # BGR → RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))   # Resize to 224x224
    img = img.astype("float32") / 255.0           # Normalize
    img = np.expand_dims(img, axis=0)             # Add batch dimension
    return img


# ─────────────────────────────────────────────
# 6. PREDICTION
# ─────────────────────────────────────────────
def predict_breed(image_path, model_path=MODEL_PATH, labels_path="class_labels.npy", top_k=3):
    """Predict dog breed from an image."""
    print(f"\n[INFO] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    idx_to_class = np.load(labels_path, allow_pickle=True).item()

    img = preprocess_image(image_path)
    preds = model.predict(img)[0]

    top_indices = np.argsort(preds)[::-1][:top_k]
    results = [(idx_to_class[i].replace("_", " ").title(), round(float(preds[i]) * 100, 2))
               for i in top_indices]

    print("\n🐾 Prediction Results:")
    print("─" * 35)
    for rank, (breed, confidence) in enumerate(results, 1):
        print(f"  #{rank}  {breed:<25} {confidence:.2f}%")
    print("─" * 35)

    # Display image with top prediction
    img_display = cv2.imread(image_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_display)
    plt.axis("off")
    plt.title(f"Predicted: {results[0][0]}  ({results[0][1]}%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150)
    plt.show()

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dog Breed Prediction System")
    parser.add_argument("--mode",  choices=["train", "predict"], required=True,
                        help="train: train the model | predict: predict from image")
    parser.add_argument("--image", type=str, help="Path to image for prediction")
    parser.add_argument("--dataset", type=str, default=DATASET_DIR, help="Dataset directory")
    args = parser.parse_args()

    if args.mode == "train":
        train_model(dataset_dir=args.dataset)
    elif args.mode == "predict":
        if not args.image:
            print("[ERROR] Please provide --image path for prediction.")
        else:
            predict_breed(args.image)
