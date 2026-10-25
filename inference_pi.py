"""
Run nutrient-deficiency inference on Raspberry Pi 3B+ with Pi Camera.

Expects in --model-dir:
    model.tflite
    labels.txt
    training_config.json (optional; supplies img_size from training)

Camera: picamera2 on Raspberry Pi OS Bookworm (libcamera).
Desktop test without hardware: pass --image path/to/leaf.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Prefer tflite_runtime on Pi (lightweight). Full TensorFlow works as fallback on PC.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite  # type: ignore[attr-defined]

DEFAULT_IMG_SIZE = (128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi leaf nutrient inference")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts"),
        help="Folder containing model.tflite and labels.txt",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image file (skip camera; useful on PC)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Dummy inferences before timing (stabilizes first-run latency)",
    )
    return parser.parse_args()


def load_training_config(model_dir: Path) -> dict:
    path = model_dir / "training_config.json"
    if not path.is_file():
        return {"img_size": list(DEFAULT_IMG_SIZE)}
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    return [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_tflite_interpreter(model_path: Path) -> tflite.Interpreter:
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def preprocess_bgr_frame(
    frame_bgr: np.ndarray,
    img_size: tuple[int, int],
    input_details: dict,
) -> np.ndarray:
    """
    Resize and pack a batch tensor for TFLite.

    Training models include normalization (MobileNet preprocess or Rescaling),
    so we pass RGB float32 in [0, 255] — same as Keras image_dataset_from_directory.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size, interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32)
    batch = np.expand_dims(tensor, axis=0)

    expected_shape = input_details["shape"]
    if list(batch.shape) != list(expected_shape):
        batch = np.reshape(batch, expected_shape)
    return batch


def capture_from_picamera2() -> np.ndarray:
    """Capture one still frame from the Pi Camera Module (picamera2)."""
    from picamera2 import Picamera2

    picam = Picamera2()
    config = picam.create_still_configuration(main={"size": (640, 480)})
    picam.configure(config)
    picam.start()
    time.sleep(0.5)  # allow auto-exposure to settle
    frame = picam.capture_array()
    picam.stop()
    picam.close()

    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def run_inference(
    interpreter: tflite.Interpreter,
    input_tensor: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details["index"])[0]
    if probs.dtype != np.float32:
        probs = probs.astype(np.float32)
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    return class_id, confidence, probs


def main() -> None:
    args = parse_args()
    model_path = args.model_dir / "model.tflite"
    labels_path = args.model_dir / "labels.txt"

    if not model_path.is_file():
        print(f"Error: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    config = load_training_config(args.model_dir)
    img_size_raw = config.get("img_size", list(DEFAULT_IMG_SIZE))
    img_size = (int(img_size_raw[0]), int(img_size_raw[1]))

    labels = load_labels(labels_path)
    interpreter = load_tflite_interpreter(model_path)
    input_details = interpreter.get_input_details()[0]

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            sys.exit(1)
        print(f"Using image file: {args.image}")
    else:
        print("Capturing from Pi Camera (picamera2)...")
        frame = capture_from_picamera2()

    input_tensor = preprocess_bgr_frame(frame, img_size, input_details)

    for _ in range(args.warmup):
        run_inference(interpreter, input_tensor)

    t0 = time.perf_counter()
    class_id, confidence, probs = run_inference(interpreter, input_tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label = labels[class_id] if class_id < len(labels) else f"unknown_index_{class_id}"

    print("-" * 40)
    print(f"Predicted class: {label}")
    print(f"Confidence:      {confidence * 100:.2f}%")
    print(f"Inference time:  {elapsed_ms:.1f} ms")
    print("-" * 40)
    print("Top classes:")
    top_k = np.argsort(probs)[::-1][: min(5, len(labels))]
    for idx in top_k:
        name = labels[idx] if idx < len(labels) else str(idx)
        print(f"  {name}: {probs[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()
