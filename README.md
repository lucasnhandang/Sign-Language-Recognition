# Sign Language Recognition with Few-Shot Learning

This is my **mini-project to practice Computer Vision (CV), Machine Learning (ML), and Deep Learning (DL)**. It provides **real-time recognition of American Sign Language (ASL) alphabets** using **EfficientNet-B0** as the backbone.  
The system supports **few-shot learning**, allowing users to define and recognize custom gestures with just 3–5 sample images.

## Features

- Recognition 26 ASL alphabets (A–Z)
- Uses EfficientNet-B0 with PyTorch as the backbone for feature extraction
- Few-shot learning capability, add new gestures without retraining
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
python app.py --model models/final_model.pth --few_shot models/few_shot_data.pkl
```

## Using the Application

### ASL Recognition
When the app is running, it has recognition mode by default, using the webcam to detect and classify ASL gestures in real time.

### Few-shot Learning Mode
To add a new custom gesture:

1. Press 'n' while the app is running
2. Enter the label for the new gesture when prompted (e.g., "UwU")
3. Perform the gesture within the green rectangle
4. Press 'c' to capture a sample (repeat 3–5 times)
5. Once enough samples are captured, the app automatically returns to recognition mode and can now recognize your custom gesture

## How Few-shot Learning Works?

This project uses a two-stage classification process:

1. **Feature Extraction**: EfficientNet-B0 backbone, pre-trained on ImageNet and fine-tuned on the ASL dataset, extracts high-level features from hand gestures.

2. **Classification**:
   - Standard ASL: A fully-connected layer classifies into one of the 26 alphabet classes.
   - Custom Gestures: A K-Nearest Neighbors (KNN) classifier compares the feature embeddings with stored custom gesture samples.

When a new gesture is added:
1. The app captures 3–5 feature embeddings.
2. These are added to the KNN model without retraining the CNN.
3. During inference, predictions from both the standard classifier and the few-shot classifier are compared based on confidence scores.