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

These setup commands target **Raspberry Pi OS Bookworm 64-bit** (Python 3.11)
with a CSI camera such as Camera Module v1. Raspberry Pi OS Trixie ships
Python 3.13, which is not supported by the available TFLite runtime wheels.
Install Bookworm 64-bit if your Pi is currently running Trixie. Install camera
and computer-vision packages from Raspberry Pi OS;
Picamera2 depends on the OS camera stack and should not be installed from PyPI.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-venv rpicam-apps
```

Connect the camera ribbon cable to the Pi 3B+'s CSI connector, then verify the
camera before running Python:

```bash
rpicam-hello --timeout 5000
```

Copy the trained `artifacts/` folder from the training computer to the project
directory on the Pi. It must contain `model.tflite`, `labels.txt`, and
`training_config.json`. Confirm Python 3.11, then create a virtual environment
that can see the camera and OpenCV packages installed by apt. Install Google's
LiteRT interpreter, which provides the same `Interpreter` API used by this
project:

```bash
python3 --version
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-pi.txt
```

Run one capture and classify it:

```bash
python inference_pi.py --model-dir artifacts
```

The script captures one frame with **Picamera2**, resizes it to the model's
configured input size, runs `model.tflite`, and prints the predicted class,
confidence, and inference time. To test the model without using the camera,
pass a saved image with `--image path/to/leaf.jpg`.

If `rpicam-hello` cannot find the camera, check that the ribbon cable is fully
seated and correctly oriented at both ends, then reboot and try again. If pip
cannot install LiteRT, confirm the Pi is running 64-bit Bookworm and the
virtual environment uses Python 3.11 (`python --version`).

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
