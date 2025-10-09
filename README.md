# 🐍 Snake Tracking Motion – Prediction Guide

This document explains how to **run predictions** on images or videos using the trained YOLO model (`vu.pt`, `last.pt`, or `best.pt`) with integrated **ByteTrack object tracking**.

---

## 📦 1. Prerequisites

Make sure your environment already has the following:

### 🧠 Core dependencies
```bash
pip install ultralytics==8.3.207
pip install pillow
pip install moviepy
pip install opencv-python
pip install torch torchvision torchaudio
```

If tracking fails due to ByteTrack issues, update Ultralytics:
```bash
pip install -U ultralytics
```

---

## 🎯 2. Run the Prediction Script

The prediction script supports **images and videos**.

### ▶️ Option 1: Run directly in terminal
```bash
python train/Predict.py
```

- A file selection dialog will appear.
- You can choose an image (`.jpg`, `.png`, `.webp`, etc.) or video (`.mp4`, `.avi`, `.mov`, etc.).
- If the selected file is `.webp`, it will automatically be converted to `.jpg`.

### ⚙️ Internally, the script does:
1. Loads your trained model (example: `../models/vu.pt`).
2. Runs prediction or tracking (depending on file type).
3. Uses **ByteTrack** for object ID tracking.
4. Saves results (images or processed video) into the output folder.
5. Converts `.avi` output video to `.mp4` automatically.

---

## 🧩 3. Code Structure

```python
# Main flow of Predict.py

from ultralytics import YOLO
from tkinter import Tk, filedialog
from PIL import Image
from moviepy.editor import VideoFileClip
import os

# Apply ByteTrack hotfix
# ... (auto fixes fuse_score error)

# Open file dialog
file_path = filedialog.askopenfilename(...)

# Handle webp conversion
if file_path.lower().endswith(".webp"):
    Image.open(file_path).convert("RGB").save(...)

# Load trained model
model = YOLO("../models/vu.pt")

# Run tracking or detection
results = model.track(
    source=file_path,
    conf=0.1,
    save=True,
    show=True,
    tracker="bytetrack.yaml"
)

# Convert AVI → MP4
# ... (auto handled by moviepy)
```

---

## 📁 4. Output Directory

The results (bounding boxes, labels, and tracked videos) are saved under:
```
runs/detect/predict/
```
or automatically generated folders such as:
```
runs/track/predictX/
```

You’ll find:
- `labels.txt` (bounding box info)
- processed image(s) or video(s)
- optionally, `predictX/labels/` if `save_txt=True` is enabled

---

## 🎞️ 5. Example Usage

### 🖼️ Predict single image
```
Input: snake_1.jpg  
→ Output: runs/detect/predict/snake_1.jpg
```

### 🎬 Predict & track in a video
```
Input: snake1.mp4  
→ Output: runs/track/predict/snake1.mp4  
→ Object IDs are shown using ByteTrack
```

Example terminal log:
```
📸 File đã chọn: C:/Users/ThanhVu/Desktop/TrackingMotionSnake/input/snake1.mp4
✅ ByteTrack fuse_score hotfix applied!
✅ Dự đoán xong! Kết quả lưu trong thư mục: runs/track/predict
✅ Video MP4 đã lưu tại: runs/track/predict/snake1.mp4
```

---

## 💡 6. Notes

- Set `conf=0.1` to adjust confidence threshold.
- If you only want detection (no tracking), replace:
  ```python
  results = model.track(...)
  ```
  with:
  ```python
  results = model.predict(...)
  ```

- If ByteTrack crashes, try:
  ```bash
  pip install -U ultralytics lap
  ```
- You can also switch to **BoT-SORT** (more stable for videos):
  ```python
  tracker="botsort.yaml"
  ```

---

## 🧠 7. Typical File Layout

```
SnakeTrackingMotion/
├── models/
│   ├── vu.pt
│   ├── last.pt
│   └── best.pt
├── train/
│   ├── Predict.py
│   ├── ResumeTrain.py
│   └── Train.py
├── input/
│   ├── snake1.mp4
│   └── test.jpg
└── runs/
    ├── detect/
    └── track/
```

---

## 🧾 8. Credits

- **YOLOv8 / YOLO11 by Ultralytics**
- **ByteTrack** for object tracking
- **MoviePy** for AVI→MP4 video conversion  
- Developed by **Thanh Vũ – SnakeTrackingMotion Project**
