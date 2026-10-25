# Soil Nutrient Deficiency Detection (Leaf Images)

Lightweight image classifier for **soil nutrient deficiency** from leaf photos. Train on a **PC or Google Colab**, deploy on **Raspberry Pi 3B+** with a **Pi Camera** using TensorFlow Lite.

**Dataset:** [Mendeley – Leaf nutrient deficiency images](https://data.mendeley.com/datasets/7vpdrbdkd4/2)

## Project layout

```
soil-nutrient-leaf-classifier/
├── data/                  # you create: one subfolder per class
├── artifacts/             # created by training (model + labels)
├── train_model.py         # train on PC / Colab
├── inference_pi.py        # run on Raspberry Pi (or --image on PC)
├── inference_webcam.py    # PC USB webcam: auto-capture + classify
├── requirements.txt       # training (+ optional local inference test)
├── requirements-pi.txt    # Pi-only dependencies
└── README.md
```

## 1. Prepare the dataset

1. Download and extract the [Mendeley archive](https://data.mendeley.com/datasets/7vpdrbdkd4/2).
2. Copy or reorganize images under `data/` with **one folder per class**. Folder names become labels in `artifacts/labels.txt`.

Example:

```
data/
  Healthy/
  Nitrogen_Deficiency/
  Phosphorus_Deficiency/
  Potassium_Deficiency/
  ...
```

Use short folder names without spaces (replace spaces with underscores) so labels are easy to read on the Pi.

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp` (standard Keras loader).

## 2. Train on PC or Colab

### Local PC

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

python train_model.py --data-dir data --output-dir artifacts --epochs 25
```

### Google Colab (sketch)

Upload `train_model.py`, mount Drive, and point `--data-dir` at your extracted dataset:

```python
!pip install -q "tensorflow>=2.15,<2.20" matplotlib
!python train_model.py --data-dir /content/drive/MyDrive/leaf_dataset --output-dir /content/artifacts --epochs 25
```

Then download `artifacts/` (`model.tflite`, `labels.txt`, `training_config.json`).

### Training options

| Flag | Description |
|------|-------------|
| `--architecture mobilenet` | MobileNetV2 head (default, better accuracy) |
| `--architecture compact_cnn` | Smaller custom CNN (faster on Pi 3B+) |
| `--fine-tune` | Short MobileNetV2 fine-tune after head training |
| `--epochs`, `--batch-size`, `--val-split` | Standard training hyperparameters |

### Training outputs (`artifacts/`)

| File | Purpose |
|------|---------|
| `model.keras` | Full Keras model |
| `model.tflite` | TFLite model for the Pi (DEFAULT optimizations) |
| `labels.txt` | Class names, one per line |
| `training_history.png` | Accuracy / loss curves |
| `training_config.json` | Image size and metadata for inference |

Copy `model.tflite`, `labels.txt`, and `training_config.json` to the Pi (SCP, USB, etc.).

## 3. Run inference on Raspberry Pi 3B+

On **Raspberry Pi OS (Bookworm)** with the camera enabled:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-pi.txt
# If tflite_runtime has no wheel for your Python version, use a matching wheel
# from TensorFlow’s Pi builds, or: pip install tensorflow (heavier fallback)

python inference_pi.py --model-dir artifacts
```

The script captures one frame with **picamera2**, resizes to 128×128, runs **model.tflite**, and prints the predicted class and confidence.

### PC webcam (USB camera)

Opens the default webcam, counts down 3 seconds so you can align a leaf, then captures and classifies:

```bash
pip install opencv-python-headless tensorflow   # or tflite-runtime if available
python inference_webcam.py --model-dir artifacts
```

Options: `--camera 1` (second camera), `--countdown 5`, `--save capture.jpg`, `--no-window` (no GUI).

### Test with a saved image (PC or Pi)

```bash
pip install -r requirements.txt   # includes opencv + optional tensorflow fallback
python inference_pi.py --model-dir artifacts --image path/to/leaf.jpg
```

Example output:

```
Predicted class: Nitrogen_Deficiency
Confidence:      87.42%
Inference time:  420.3 ms
```

## Hardware notes (Pi 3B+)

- Input size is **128×128** (fixed in training for edge latency).
- Prefer **`compact_cnn`** if inference is too slow; **MobileNetV2** is more accurate but heavier.
- Use even lighting and a plain background when photographing leaves.

## License

Use the Mendeley dataset according to its terms. Application code in this repository is provided as-is for education and prototyping.
