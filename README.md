# Automatic Number Plate Recognition System



![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7-orange)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15-green)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-lightgrey)

An intelligent system for automatic license plate detection and recognition using computer vision and machine learning, featuring both real-time camera processing and video file analysis.

## Key Features

- **Multi-Source Input**
  - Real-time camera feed processing
  - Pre-recorded video file support
  - Adaptive frame rate handling

- **Advanced Detection**
  - YOLOv4-tiny for high-accuracy plate localization
  - Contour-based fallback detection
  - Aspect ratio validation (2:1 to 6:1)

-  **OCR Processing**
  - Tesseract OCR with custom whitelist (`A-Z0-9`)
  - Confidence-based result filtering
  - Text normalization

- **Data Management**
  - Automatic CSV logging (timestamp, plate, confidence)
  - Detected plate image archival
  - Exportable logs

- 📱 **User-Friendly UI**
  - Live video display with aspect ratio preservation
  - Detection statistics dashboard
  - Interactive log viewer

