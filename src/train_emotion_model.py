import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory

# ==========================================================
# Configuration
# ==========================================================

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_BEST = os.path.join(MODELS_DIR, "emotion_model.h5")
MODEL_FINAL = os.path.join(MODELS_DIR, "emotion_final.h5")
LABELS_FILE = os.path.join(MODELS_DIR, "emotion_labels.json")
PLOT_FILE = os.path.join(MODELS_DIR, "training_history.png")

os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================================
# Dataset Loader
# ==========================================================

def load_datasets():

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training folder not found:\n{TRAIN_DIR}"
        )

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(
            f"Testing folder not found:\n{TEST_DIR}"
        )

    print("=" * 60)
    print("Loading FER2013 Folder Dataset")
    print("=" * 60)

    train_ds = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical"
    )

    val_ds = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.20,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical"
    )

    test_ds = image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    print("\nDetected Classes:")
    for i, c in enumerate(class_names):
        print(f"{i} -> {c}")

    with open(LABELS_FILE, "w") as f:
        json.dump(class_names, f, indent=4)

    normalization = layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    print("\nDataset Loaded Successfully")

    return train_ds, val_ds, test_ds, class_names
# ==========================================================
# CNN Model
# ==========================================================

def build_model(num_classes):

    model = models.Sequential([

        layers.Input(shape=(48, 48, 1)),

        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.5),

        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation="softmax")

    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# Plot Training History
# ==========================================================

def plot_history(history):

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy")
    plt.legend(["Train","Validation"])

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss")
    plt.legend(["Train","Validation"])

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()


# ==========================================================
# Training
# ==========================================================

def train():

    train_ds, val_ds, test_ds, class_names = load_datasets()

    model = build_model(len(class_names))

    model.summary()

    checkpoint = callbacks.ModelCheckpoint(
        MODEL_BEST,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    earlystop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            earlystop,
            reduce_lr
        ]
    )

    print("\nEvaluating on Test Dataset...\n")

    loss, acc = model.evaluate(test_ds)

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Loss     : {loss:.4f}")

    model.save(MODEL_FINAL)

    plot_history(history)

    print("\nTraining Completed Successfully.")

    print("\nFiles Saved:")

    print(MODEL_BEST)
    print(MODEL_FINAL)
    print(LABELS_FILE)
    print(PLOT_FILE)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("="*55)
    print(" ATTENDANCE SYSTEM - Emotion Model Training")
    print("="*55)

    train()