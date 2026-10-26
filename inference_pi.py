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

# Prefer LiteRT on Pi, with legacy TFLite Runtime and full TensorFlow fallbacks.
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite  # type: ignore[attr-defined]

DEFAULT_IMG_SIZE = (128, 128)
PREVIEW_WINDOW = "Leaf classifier - position image, Esc to cancel"
RESULT_WINDOW = "Leaf classifier - press any key to close"


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
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Seconds to preview before each capture attempt",
    )
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default=Path("feedback_images"),
        help="Folder for captured images labeled with the confirmed class",
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


def is_valid_leaf_frame(frame: np.ndarray) -> bool:
    """Reject blank images and images without enough visible leaf color/detail."""
    if frame is None or frame.size == 0:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green_mask = (h >= 35) & (h <= 90) & (s >= 35) & (v >= 40)
    if float(np.mean(green_mask)) < 0.08:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 18.0:
        return False

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return float(np.mean(edges > 0)) > 0.01


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


def load_face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's frontal-face cascade from its packaged data directory."""
    cascade_classifier = getattr(cv2, "CascadeClassifier", None)
    if cascade_classifier is None:
        raise RuntimeError(
            "This OpenCV build does not include face detection. Install the "
            "Raspberry Pi OS packages with: sudo apt install python3-opencv opencv-data"
        )

    cascade_name = "haarcascade_frontalface_default.xml"
    candidate_paths = [
        Path("/usr/share/opencv4/haarcascades") / cascade_name,
        Path("/usr/share/opencv/haarcascades") / cascade_name,
    ]
    cv2_data = getattr(cv2, "data", None)
    if cv2_data is not None:
        cascade_dir = getattr(cv2_data, "haarcascades", None)
        if cascade_dir:
            candidate_paths.insert(0, Path(cascade_dir) / cascade_name)

    for path in candidate_paths:
        detector = cascade_classifier(str(path))
        if not detector.empty():
            return detector

    raise FileNotFoundError(
        "OpenCV face-detection data is missing. Install it with: "
        "sudo apt install opencv-data"
    )


def contains_human_face(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
) -> bool:
    """Detect a face in the central image area, avoiding people beside a phone."""
    height, width = frame.shape[:2]
    center = frame[height // 5 : height - height // 5, width // 5 : width - width // 5]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return len(faces) > 0


def capture_from_picamera2(countdown_seconds: int) -> np.ndarray:
    """Preview the Pi Camera and retry captures until a valid leaf image is shown."""
    from picamera2 import Picamera2

    face_detector = load_face_detector()
    picam = Picamera2()
    started = False
    try:
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        picam.configure(config)
        picam.start()
        started = True
        time.sleep(1)
        countdown_start = time.monotonic()
        retry_message_until = 0.0
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        while True:
            frame = picam.capture_array()
            now = time.monotonic()
            display = frame.copy()

            if now < retry_message_until:
                message = "show a proper image"
                color = (0, 0, 255)
            else:
                if retry_message_until:
                    countdown_start = now
                    retry_message_until = 0.0

                elapsed = now - countdown_start
                remaining = max(0, countdown_seconds - int(elapsed))
                message = f"Capturing in {remaining} s - show a leaf photo"
                color = (0, 255, 0)

                if elapsed >= countdown_seconds:
                    if not is_valid_leaf_frame(frame) or contains_human_face(
                        frame, face_detector
                    ):
                        print("Invalid capture: show a proper image", file=sys.stderr)
                        retry_message_until = now + 2
                        message = "show a proper image"
                        color = (0, 0, 255)
                    else:
                        return frame

            cv2.putText(
                display,
                message,
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(PREVIEW_WINDOW, display)
            if cv2.waitKey(1) & 0xFF == 27:
                raise SystemExit("Capture cancelled by user")
    finally:
        if started:
            picam.stop()
        picam.close()
        cv2.destroyAllWindows()


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


def save_confirmed_feedback(
    frame: np.ndarray,
    labels: list[str],
    feedback_dir: Path,
) -> Path | None:
    """Ask for the true class and save the frame into that class's folder."""
    print("\nWas the prediction correct? Select the image's true class to save it:")
    for index, label in enumerate(labels, start=1):
        print(f"  {index}: {label}")
    print("  0: discard this image")

    while True:
        choice = input("True class number (0 to discard): ").strip()
        if choice == "0":
            print("Image discarded; no feedback image saved.")
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(labels):
            label = labels[int(choice) - 1]
            break
        print(f"Enter a number from 0 to {len(labels)}.")

    if (
        not label
        or label in {".", ".."}
        or Path(label).name != label
        or "/" in label
        or "\\" in label
    ):
        raise ValueError(f"Class label cannot be used as a folder name: {label!r}")

    class_dir = feedback_dir / label
    class_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = class_dir / f"{timestamp}_{time.time_ns() % 1_000_000_000:09d}.jpg"
    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f"Could not save feedback image: {image_path}")
    print(f"Saved confirmed {label} image: {image_path}")
    return image_path


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
        if args.countdown < 1:
            print("Error: --countdown must be at least 1 second", file=sys.stderr)
            sys.exit(2)
        print(f"Starting Pi Camera preview ({args.countdown}-second countdown)...")
        frame = capture_from_picamera2(args.countdown)

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

    if not args.image:
        cv2.imshow(RESULT_WINDOW, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_confirmed_feedback(frame, labels, args.feedback_dir)


if __name__ == "__main__":
    main()
