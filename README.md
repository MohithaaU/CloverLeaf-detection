#  Clover Leaf Detection using YOLOv8

A Computer Vision project that detects and classifies clover leaves using YOLOv8 object detection. The project focuses on identifying rare 4-leaf clovers among normal 3-leaf clovers and supports both image-based and real-time webcam detection.

---

## Project Overview

This project was developed to explore practical Computer Vision workflows including:

- Object Detection using YOLOv8
- Dataset preprocessing and augmentation
- Transfer Learning
- Model evaluation using mAP, Precision, and Recall
- Real-time inference using OpenCV
- Performance comparison across multiple dataset versions

---

## Features

✅ Detects clover leaves using YOLOv8

✅ Real-time webcam inference

✅ Bounding box visualization

✅ Multiple dataset experiments

✅ Performance evaluation and comparison

---

## Tech Stack

| Tool | Function |
|--------|--------|
| Python | Core programming language |
| YOLOv8 (Ultralytics) | Object detection framework |
| OpenCV | Real-time webcam processing |
| PyTorch | Deep learning backend |
| Roboflow | Dataset management and augmentation |
| Google Colab | GPU-based model training |
| VS Code | Local development |
| Git & GitHub | Version control |

---

## Dataset Versions

### Version 1
- Total Images: 376
- Classes: 1 (4-leaf clover)
- mAP50: 0.754

### Version 2
- Total Images: 3355
- Classes: 4
- mAP50: 0.404

### Version 3
- Total Images: 1403
- Classes: 4
- mAP50: 0.333

### Version 4
- Total Images: 1194
- Classes: 4
- mAP50: 0.311

---

## Model Performance

| Version | Dataset Size | Classes | mAP50 | Precision | Recall |
|----------|----------|----------|----------|----------|----------|
| V1 | 376 | 1 | 0.754 | 0.733 | 0.684 |
| V2 | 3355 | 4 | 0.404 | 0.465 | 0.404 |
| V3 | 1403 | 4 | 0.333 | 0.395 | 0.439 |
| V4 | 1194 | 4 | 0.311 | 0.317 | 0.474 |

---

## Key Learnings

- More data does not always improve model performance.
- Dataset quality is often more important than dataset size.
- Class imbalance significantly affects detection accuracy.
- Label consistency is critical in multi-class object detection.
- Transfer learning greatly reduces training time.
- Fine-grained object detection remains challenging even with pretrained models.

---

## Real-Time Webcam Detection

The project includes a webcam-based detection system using OpenCV.

Workflow:

1. Capture webcam frame
2. Pass frame into YOLOv8 model
3. Generate bounding boxes
4. Display predictions in real time

---

## Challenges Faced

- Class imbalance between 3-leaf and 4-leaf clovers
- Limited availability of 4-leaf clover images
- Label inconsistency across datasets
- Overlapping bounding boxes in dense clover patches
- Google Colab runtime limitations

---

## Future Improvements

- Collect custom real-world clover datasets
- Merge inconsistent labels
- Improve class balance
- Experiment with larger YOLOv8 models
- Hyperparameter tuning
- Deploy as a web application

---

## Repository Structure

```text
CloverLeaf-detection/
│
├── train.py
├── detect.py
├── webc.py
├── eval.py
├── Clovercode.ipynb
├── .gitignore
└── README.md
