# Sign Language Recognition

This mini-project provides **real-time recognition of American Sign Language (ASL) alphabet letters** from webcam input using deep learning models. It uses pre-trained versions of **EfficientNet-B0** and **ConvNeXt-Tiny**, fine-tuned on the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

> Pre-trained weights are provided and no training is required.  
>  
> This project was built as a hands-on exercise to strengthen my understanding of **Computer Vision** and **Deep Learning**, especially in the areas of **image classification**, **model fine-tuning**, and **real-time inference using webcam data**.

---
## Features

- Recognizes 26 ASL letters (A–Z) from hand gesture images
- Real-time prediction using a computer webcam
- Supports two deep learning models:
  - **EfficientNet-B0** (lightweight & fast)
  - **ConvNeXt-Tiny** (SOTA)
- User-friendly CLI for model selection

---
## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/lucasnhandang/Sign-Language-Recognition.git
cd Sign-Language-Recognition
```

### 2. Install dependencies
Make sure you are using Python 3.8+, then run:
```bash
pip install -r requirements.txt
```

### 3. Run ```app.py```!

---
## Dataset
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Over 87,000 RGB images (200x200)
- 3000+ samples per letter (A–Z)

---
## Project Structure
```
Sign-Language-Recognition/
├── models/                 # Pretrained .pth files
├── app.py                  
├── requirements.txt        # Dependency list
├── utils.py                
└── README.md               
```

---
## License
This project is open-source under the MIT License.