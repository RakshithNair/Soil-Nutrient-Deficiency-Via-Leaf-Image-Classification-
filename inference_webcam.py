"""
Live webcam capture + nutrient-deficiency inference (PC / laptop).

Opens the default USB webcam, warms up auto-exposure, captures one frame,
runs model.tflite, and prints results. Optional preview countdown and result window.

Usage:
    python inference_webcam.py --model-dir artifacts
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from inference_pi import (
    load_labels,
    load_tflite_interpreter,
    load_training_config,
    preprocess_bgr_frame,
    run_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam leaf nutrient inference")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts"),
        help="Folder with model.tflite and labels.txt",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (0 = default camera)",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds of live preview before auto-capture (0 = capture immediately)",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open GUI windows (headless: capture and print only)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the captured frame (e.g. capture.jpg)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Dummy inferences before timed run",
    )
    return parser.parse_args()


def open_webcam(camera_index: int) -> cv2.VideoCapture:
    """Open webcam; on Windows prefer DirectShow for faster open."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera_index}. "
            "Try --camera 1 or close other apps using the camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def capture_frame(
    cap: cv2.VideoCapture,
    countdown_sec: int,
    show_window: bool,
) -> np.ndarray:
    """
    Read frames until countdown ends, then return the last frame.
    Press Esc during preview to abort.
    """
    settle = 15
    for _ in range(settle):
        cap.read()

    if countdown_sec <= 0 or not show_window:
        for _ in range(5):
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read from webcam")
        return frame

    window = "Leaf classifier — align leaf, auto-capture soon (Esc to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    start = time.time()
    frame: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read from webcam")

        elapsed = time.time() - start
        remaining = max(0, countdown_sec - int(elapsed))
        display = frame.copy()
        cv2.putText(
            display,
            f"Capturing in {remaining + 1} s — hold leaf in frame",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled by user")
        if elapsed >= countdown_sec:
            break

    cv2.destroyAllWindows()
    return frame


def draw_result(frame: np.ndarray, label: str, confidence: float) -> np.ndarray:
    out = frame.copy()
    text = f"{label} ({confidence * 100:.1f}%)"
    cv2.rectangle(out, (0, 0), (out.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def print_results(
    label: str,
    confidence: float,
    elapsed_ms: float,
    labels: list[str],
    probs: np.ndarray,
) -> None:
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


def contains_human_face(frame: np.ndarray) -> bool:
    """Reject frames where a human face/skin region is visible."""
    if frame is None or frame.size == 0:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    lower_skin = ((h >= 0) & (h <= 25)) | ((h >= 170) & (h <= 180))
    skin_mask = lower_skin & (s >= 25) & (v >= 40)
    skin_ratio = float(np.mean(skin_mask > 0))
    return skin_ratio > 0.03


def is_valid_leaf_frame(frame: np.ndarray) -> bool:
    """Return True only for frames that look like a leaf scene, not a person or blank background."""
    if frame is None or frame.size == 0:
        return False
    if contains_human_face(frame):
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if h.size == 0 or s.size == 0 or v.size == 0:
        return False

    green_mask = ((35 <= h) & (h <= 90) & (s >= 35) & (v >= 40)).astype(np.uint8)
    green_ratio = float(np.mean(green_mask > 0))
    if green_ratio < 0.08:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std_dev = float(gray.std())
    if std_dev < 18.0:
        return False

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_ratio = float(np.mean(edges > 0))
    return edge_ratio > 0.01 and green_ratio > 0.08


def main() -> None:
    args = parse_args()
    model_path = args.model_dir / "model.tflite"
    labels_path = args.model_dir / "labels.txt"

    if not model_path.is_file():
        print(f"Error: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    config = load_training_config(args.model_dir)
    img_size_raw = config.get("img_size", [128, 128])
    img_size = (int(img_size_raw[0]), int(img_size_raw[1]))

    labels = load_labels(labels_path)
    interpreter = load_tflite_interpreter(model_path)
    input_details = interpreter.get_input_details()[0]

    print(f"Opening webcam (index {args.camera})...")
    cap = open_webcam(args.camera)
    try:
        frame = capture_frame(cap, args.countdown, show_window=not args.no_window)
    finally:
        cap.release()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), frame)
        print(f"Saved capture: {args.save}")

    if not is_valid_leaf_frame(frame):
        msg = "show a proper image"
        print(msg, file=sys.stderr)
        if args.no_window:
            return

        display = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            display,
            msg,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Leaf classifier", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    input_tensor = preprocess_bgr_frame(frame, img_size, input_details)
    for _ in range(args.warmup):
        run_inference(interpreter, input_tensor)

    t0 = time.perf_counter()
    class_id, confidence, probs = run_inference(interpreter, input_tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label = labels[class_id] if class_id < len(labels) else f"unknown_index_{class_id}"
    print_results(label, confidence, elapsed_ms, labels, probs)

    if not args.no_window:
        result_view = draw_result(frame, label, confidence)
        cv2.imshow("Result — press any key to close", result_view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
