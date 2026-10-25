from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite  # type: ignore[attr-defined]

MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "model.tflite"
LABELS_PATH = MODEL_DIR / "labels.txt"
DEFAULT_IMG_SIZE = (128, 128)

app = FastAPI(title="Leaf Nutrient Classifier")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def load_labels() -> list[str]:
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")
    labels = [
        line.strip() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return labels


def load_tflite_interpreter():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


interpreter = load_tflite_interpreter()
labels = load_labels()
input_details = interpreter.get_input_details()[0]


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, DEFAULT_IMG_SIZE, interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32)
    batch = np.expand_dims(tensor, axis=0)
    expected_shape = input_details["shape"]
    if list(batch.shape) != list(expected_shape):
        batch = np.reshape(batch, expected_shape)
    return batch


def predict_from_frame(frame_bgr: np.ndarray) -> dict:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Frame is empty")

    tensor = preprocess_frame(frame_bgr)
    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    probs = interpreter.get_tensor(output_details["index"])[0].astype(np.float32)
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    prediction = {
        "class_name": labels[class_id] if class_id < len(labels) else f"unknown_index_{class_id}",
        "confidence": round(confidence * 100, 2),
        "top_classes": [
            {
                "label": labels[i] if i < len(labels) else str(i),
                "confidence": round(float(probs[i]) * 100, 2),
            }
            for i in np.argsort(probs)[::-1][: min(5, len(labels))]
        ],
    }
    return prediction


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(None)):
    if file is None:
        raise HTTPException(status_code=400, detail="No image uploaded")

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode uploaded image")
        result = predict_from_frame(frame)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
