# Object Detection Pipeline — Bottle · Tin Can · Dice · Ball
### YOLOv11n + Open Images v4 Dataset

An end-to-end pipeline for training a custom YOLOv11 object detector using freely available Open Images v4 data.

---

## AI Model Specification

| Detail | Info |
|--------|------|
| **Model** | Ultralytics YOLOv11 nano |
| **Classes** | 4 (Bottle, Tin can, Dice, Ball) |
| **Training Framework** | NVIDIA CUDA |
| **Training Hardware** | NVIDIA GeForce GTX 1650 Mobile |
| **Target Hardware** | Raspberry Pi 5 |
| **Dataset** | Google Open Images v4 (OIDv4 ToolKit) |
| **Dataset Size** | ~3,500+ images |
| **Epochs** | 150 |
| **Image Size** | 640 |
| **Batch** | Auto (2 for 4GB VRAM) |
| **Patience (Early Stopping)** | 30 |

---

## Classes

| ID | Class | OID Folder Name |
|----|-------|----------------|
| 0 | Bottle | `Bottle` |
| 1 | Tin can | `Tin can` |
| 2 | Dice | `Dice` |
| 3 | Ball | `Ball` |

---

## Quick Start

```
Step 1 → python 1_download_dataset.py
Step 2 → python 2_convert_labels.py
Step 3 → python 3_train.py
Step 4 → python 4_detect.py
```

---

## Requirements

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) or CPU
- Git

### Install Dependencies

```bash
pip install ultralytics opencv-python Pillow pandas requests tqdm
```

### Install PyTorch

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

**GPU (NVIDIA CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Clone OIDv4 ToolKit
```bash
git clone https://github.com/EscVM/OIDv4_ToolKit.git
pip install -r OIDv4_ToolKit/requirements.txt
```

---

## Project Structure

```
yolo_object_detection/
│
├── OIDv4_ToolKit/              ← cloned from GitHub
├── OID/                        ← created by step 1 (raw images)
│   └── OID_raw/
│       ├── train/
│       │   ├── Ball/
│       │   ├── Bottle/
│       │   ├── Dice/
│       │   └── Tin can/
│       └── validation/
│
├── dataset/                    ← created by step 2 (YOLO format)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── data.yaml                   ← created by step 2
│
├── runs/detect/                ← created by step 3
│   └── bottle_tincan_dice_ball/
│       └── weights/
│           ├── best.pt         ← use this for inference
│           └── last.pt
│
├── 1_download_dataset.py
├── 2_convert_labels.py
├── 3_train.py
├── 4_detect.py
├── check_filenames.py
├── requirements.txt
└── README.md
```

---

## Step-by-Step Guide

### Step 1 — Download Dataset

Downloads images for all 4 classes from Open Images v4.

```bash
python 1_download_dataset.py
```

**Configurable variables in `1_download_dataset.py`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OIDV4_PATH` | `./OIDv4_ToolKit` | Path to OIDv4 ToolKit |
| `LIMIT` | `500` | Max images per class per split |
| `SPLITS` | `["train", "validation"]` | OID splits to download |

> ⚠️ **Windows:** The script calls `python` — if your system uses `python3`, update the command inside `1_download_dataset.py`.

---

### Step 2 — Convert Labels

Converts OID annotation format to YOLO format and builds the dataset.

```bash
python 2_convert_labels.py
```

This script will:
- Wipe the old dataset and rebuild from scratch
- Convert `ClassName left top right bottom` → `class_id x_center y_center width height`
- Normalize all values to [0, 1]
- Perform automatic train/val split
- Verify image/label filename parity
- Write `data.yaml`

> ⚠️ **Important:** `OID_ROOT` uses an absolute path. If cloning to a new machine, update it:
> ```python
> OID_ROOT = r"C:\your\path\to\OID\OID_raw"   # Windows
> OID_ROOT = "/your/path/to/OID/OID_raw"       # Linux / Mac
> ```

---

### Step 3 — Train Model

Trains YOLOv11n starting from pretrained COCO weights.

```bash
python 3_train.py
```

**Configurable variables in `3_train.py`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | `150` | Number of training epochs |
| `BATCH` | `-1` | Batch size (-1 = AutoBatch) |
| `PATIENCE` | `30` | Early stopping patience |
| `IMGSZ` | `640` | Input image resolution |

> ⚠️ **GTX 1650 / Low VRAM GPUs:** The script sets `amp=False` to prevent NaN losses caused by AMP incompatibility.

**Training output location:**
```
runs/detect/bottle_tincan_dice_ball/
├── weights/
│   ├── best.pt           ← best checkpoint
│   └── last.pt
├── results.png           ← training curves
└── confusion_matrix.png
```

**What to monitor during training:**

| Metric | Good sign |
|--------|-----------|
| `box_loss` | Steadily decreasing |
| `cls_loss` | Drops quickly and stays low |
| `mAP50` | Increasing toward 0.7+ |

---

### Step 4 — Run Inference

**Live webcam:**
```bash
python 4_detect.py
```

**Single image:**
```bash
python 4_detect.py --source path/to/image.jpg
```

**Folder of images:**
```bash
python 4_detect.py --source path/to/folder/
```

**Save results:**
```bash
python 4_detect.py --source path/to/image.jpg --save
```

**Webcam keyboard shortcuts:**
- `Q` — Quit
- `S` — Save current frame

---

## Utility Scripts

### Check Image/Label Parity
Verifies that every image has a matching label file before training.
```bash
python check_filenames.py --images dataset/images/train --labels dataset/labels/train
```

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `python3 not found` (Windows) | Windows uses `python`, not `python3` | Change `python3` → `python` in `1_download_dataset.py` |
| `Tin can` missing from dataset | `split()` breaks class names with spaces, causing silent coordinate parsing failure | Fixed — script now reads coordinates from `parts[-4:]` |
| `box_loss = 0` throughout training | AMP causes NaN losses on GTX 1650 | Fixed — script sets `amp=False` in `model.train()` |
| Webcam not opening | Privacy settings or wrong camera index | Check Windows → Settings → Privacy & Security → Camera |
| `CUDA out of memory` | Insufficient VRAM | Use `batch=-1` (AutoBatch) or reduce batch size manually |
| `yaml.scanner.ScannerError` | Labels not yet converted to YOLO format | Re-run `2_convert_labels.py` |
| `No images found` | Empty dataset folder | Run step 1 and step 2 before training |

---

## Raspberry Pi 5 Deployment

### Export Model (run on Windows/PC)
```bash
python -c "from ultralytics import YOLO; YOLO('runs/detect/bottle_tincan_dice_ball/weights/best.pt').export(format='ncnn')"
```

### Install on Pi 5
```bash
pip install ultralytics opencv-python
```

---

## What's Included in This Repo

| File | Included |
|------|----------|
| Python scripts (all 4 steps) | ✅ |
| `data.yaml` | ✅ |
| `requirements.txt` | ✅ |
| `best.pt` (trained weights) | ⚠️ Optional — add manually |
| `dataset/` (images) | ❌ Too large — generated by running step 1 & 2 |
| `OID/` (raw downloaded images) | ❌ Too large |

> If `best.pt` is included, users can run `4_detect.py` directly without training.
> If not included, users must run all 4 steps from scratch.

---

## References

- [OIDv4 ToolKit](https://github.com/EscVM/OIDv4_ToolKit)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Object Detection with YOLOv11](https://github.com/Thuta777/Object-Detection-with-YOLOv11)
- [Open Images v4 Dataset](https://storage.googleapis.com/openimages/web/index.html)

---

## License

GPL-3.0 (OIDv4 ToolKit) · AGPL-3.0 (Ultralytics YOLOv11)
