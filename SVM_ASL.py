import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import SignLanguageModel
import argparse

def main():
    print("ASL Recognition with PyTorch & Few-Shot Learning")
    print("See README.md for detailed instructions on how to use this project")
    print("\nQuick start:")
    print("1. Prepare data: python prepare_data.py --download --check_images")
    print("2. Train model: python train.py --data_dir data/processed")
    print("3. Run app: python app.py")
    print("\nFew-shot learning mode:")
    print("- Press 'n' in the app to add a new custom gesture")
    print("- Enter a label for the new gesture (e.g., 'I LOVE YOU')")
    print("- Make the gesture in the green box and press 'c' to capture (repeat 3-5 times)")
    print("- The model will automatically learn to recognize your new gesture!")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    main()