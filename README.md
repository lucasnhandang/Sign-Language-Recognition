# Sign Language Recognition with Few-Shot Learning

This is my mini-project to practice Computer Vision (CV), Machine Learning (ML), and Deep Learning (DL). 

It provides **real-time recognition** of American Sign Language (ASL) alphabets using **EfficientNet-B0** as the backbone.  

The system supports **few-shot learning**, allowing users to define and recognize custom gestures with just 5 sample images.

## Features

- Recognition 26 ASL alphabets (A–Z) + 3 more gestures in dataset
- Uses EfficientNet-B0 as the backbone for feature extraction
- Add new gestures without retraining
- Hand detection and tracking with MediaPipe
- Real-time recognition from webcam

## Getting Started

### 1. Clone repository
```bash
git clone https://github.com/yourusername/Sign-Language-Recognition.git
cd Sign-Language-Recognition
```

### 2. Install dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run app
```bash
python app.py
```

## Using the Application

When the app is running, it has recognition mode by default, using the webcam to detect and classify ASL gestures in real time.
