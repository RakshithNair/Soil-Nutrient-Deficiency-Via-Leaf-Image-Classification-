"""
Train a lightweight leaf nutrient-deficiency classifier for edge deployment.

Run on a PC or Google Colab. Expects an image folder layout (Mendeley dataset or similar):

    data/
        Healthy/
            img001.jpg
        Nitrogen_Deficiency/
            ...

Outputs (under --output-dir):
    model.keras          Full Keras model
    model.tflite         TensorFlow Lite (DEFAULT graph optimizations for Pi)
    labels.txt           One class name per line (index matches softmax output)
    training_history.png Accuracy / loss curves
    training_config.json   Image size and architecture metadata for inference_pi.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Fixed input size — matches Raspberry Pi 3B+ latency target in inference_pi.py
IMG_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 25
DEFAULT_VAL_SPLIT = 0.2
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train soil nutrient leaf classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory with one subfolder per class",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saved model and plots",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help="Fraction of images held out for validation (stratified by folder)",
    )
    parser.add_argument(
        "--architecture",
        choices=("mobilenet", "compact_cnn"),
        default="mobilenet",
        help="mobilenet: MobileNetV2 backbone; compact_cnn: small custom CNN for Pi 3B+",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze top MobileNetV2 layers for a short second training phase",
    )
    return parser.parse_args()


def build_datasets(
    data_dir: Path,
    batch_size: int,
    val_split: float,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Load images from class subfolders, resize to IMG_SIZE, split train/validation.

    Pixels are in [0, 255] as float32; normalization is applied inside the Keras model
    so the exported TFLite graph matches inference on the Pi.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir.resolve()}\n"
            "Download the Mendeley dataset and place one folder per class under data/."
        )

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def build_compact_cnn(num_classes: int) -> keras.Model:
    """Small CNN (~few hundred KB as TFLite) suitable for Raspberry Pi 3B+."""
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # Input pixels are 0–255 from the dataset pipeline
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="compact_leaf_cnn")


def build_mobilenet(num_classes: int) -> keras.Model:
    """
    MobileNetV2 with frozen ImageNet weights; train only the classification head.

    Preprocessing ([-1, 1]) is part of the graph for correct TFLite export.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mobilenet_leaf_classifier")


def fine_tune_mobilenet(model: keras.Model, fine_tune_at: int = 100) -> None:
    """Unfreeze deeper MobileNetV2 blocks for a short fine-tune phase."""
    base = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base = layer
            break
    if base is None:
        return
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False


def plot_history(history: keras.callbacks.History, output_path: Path) -> None:
    """Save side-by-side training / validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_labels(class_names: list[str], path: Path) -> None:
    """Write labels.txt — line index matches model output index."""
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def export_tflite(keras_model: keras.Model, tflite_path: Path) -> None:
    """
    Export float32 TFLite with DEFAULT optimizations (smaller / faster on Pi).

    For full integer quantization you would need a representative dataset;
    float32 + DEFAULT is a reliable baseline on Pi 3B+.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(
        args.data_dir, args.batch_size, args.val_split
    )
    num_classes = len(class_names)

    if args.architecture == "mobilenet":
        model = build_mobilenet(num_classes)
    else:
        model = build_compact_cnn(num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if args.fine_tune and args.architecture == "mobilenet":
        fine_tune_mobilenet(model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=min(10, args.epochs),
            callbacks=callbacks,
        )
        for k, v in history_ft.history.items():
            history.history.setdefault(k, []).extend(v)

    keras_path = args.output_dir / "model.keras"
    tflite_path = args.output_dir / "model.tflite"
    labels_path = args.output_dir / "labels.txt"
    plot_path = args.output_dir / "training_history.png"
    meta_path = args.output_dir / "training_config.json"

    model.save(keras_path)
    export_tflite(model, tflite_path)
    save_labels(class_names, labels_path)
    plot_history(history, plot_path)

    # Preprocessing lives inside the Keras graph; Pi feeds 0–255 RGB only
    meta_path.write_text(
        json.dumps(
            {
                "img_size": list(IMG_SIZE),
                "num_classes": num_classes,
                "class_names": class_names,
                "architecture": args.architecture,
                "input_range": "0_255_rgb",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved Keras model: {keras_path}")
    print(f"Saved TFLite model: {tflite_path} ({tflite_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved labels: {labels_path}")
    print(f"Saved training plot: {plot_path}")


if __name__ == "__main__":
    main()
