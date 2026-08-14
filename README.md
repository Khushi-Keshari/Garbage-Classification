# Waste Classification using Deep Learning

## Overview

This project implements a Convolutional Neural Network (CNN) based waste classification system using PyTorch. The model classifies waste images into 12 categories and explores the application of deep learning for automated waste sorting.

The model was trained for 50 epochs on a modified Kaggle Garbage Classification dataset and achieved 89.39% validation accuracy.

**The model predicts the type of waste from an image and can be used for smart recycling or educational applications.**


---

## Highlights

* **Dataset:** Adapted and reorganized the Kaggle Garbage Classification dataset into 12 waste categories.
* **OpenCV-Based Inference:** Implemented webcam and image-based inference using OpenCV for real-time waste classification.
* **CNN Architecture:** Designed a six-layer CNN with Batch Normalization and Dropout to improve training stability and reduce overfitting.
* **Comprehensive Evaluation:** Implemented a validation pipeline with validation accuracy, a confusion matrix, and a classification report to evaluate model performance across all 12 classes.
* **Deep Learning Implementation:** Applied CNN concepts including convolutional layers, Batch Normalization, Dropout, and optimization techniques to build an end-to-end waste classification system.

---

## Table of Contents

* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Usage Example](#usage-example)
* [Results](#results)
* [Author](#author)

---

## Project Structure

```
cnn-waste-classification-opencv-pytorch/
├── dataset/
│   ├── train/
│   │   ├── battery/
│   │   ├── biological/
│   │   ├── brown-glass/
│   │   ├── cardboard/
│   │   ├── clothes/
│   │   ├── green-glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   ├── shoes/
│   │   ├── trash/
│   │   └── white-glass/
│   ├── val/
│   │   ├── battery/
│   │   ├── biological/
│   │   ├── brown-glass/
│   │   ├── cardboard/
│   │   ├── clothes/
│   │   ├── green-glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   ├── shoes/
│   │   ├── trash/
│   │   └── white-glass/
├── saved_models/
│   └── best_model.pth
├── main.py
├── object-detection.py
├── validation-checker.py
├── validation-splitter.py
└── requirements.txt
```

* `main.py`: Trains the CNN model for 50 epochs and saves the best model to `saved_models/best_model.pth`.
* `object-detection.py`: Performs waste classification inference using webcam input or static images.
* `validation-checker.py`: Evaluates the model, computes validation accuracy, and generates a confusion matrix.
* `validation-splitter.py`: Splits the dataset into 80% training and 20% validation sets.
* `requirements.txt`: Lists dependencies like PyTorch, OpenCV, and NumPy.

---

## Setup & Installation

Follow these steps to set up and run the project:

```bash
git clone https://github.com/Khushi-Keshari/Garbage-Classification.git
cd Garbage-Classification

# Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate
# On Linux: source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt

# Train the model:
python main.py

# Perform predictions:
python object-detection.py

# Evaluate the model:
python validation-checker.py
```

---

## Dataset

The dataset is a modified version of the Kaggle Garbage Classification dataset, adapted for **12 waste categories** to perform multi-class waste classification.

Source: [Kaggle Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

The dataset contains the following waste categories:

* Battery
* Biological
* Brown Glass
* Cardboard
* Clothes
* Green Glass
* Metal
* Paper
* Plastic
* Shoes
* Trash
* White Glass

The dataset is split into:

* **Training Set:** 80% of the data.
* **Validation Set:** 20% of the data.

All images are resized to **224 × 224 pixels** to match the CNN input requirements and ensure consistent model training and inference.

---

## Model Architecture

The CNN consists of six convolutional layers followed by fully connected layers. Below is the model definition:

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),  # For 224x224 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Summary Table:**

| Layer Type         | Count | Details                            |
| ------------------ | ----- | ---------------------------------- |
| Input Layer      | 1     | RGB image input (3 channels, 224×224) |
| Conv Hidden Layers | 6     | Conv2d-BatchNorm2d-ReLU-MaxPool2d  |
| FC Hidden Layer    | 1     | Linear(512*3*3, 512)               |
| Output Layer       | 1     | Linear(512, num\_classes)          |

**Neuron Count**

* **Convolutional Layers:** Feature maps increase progressively (16, 32, 64, 128, 256, 512).
* **Fully Connected Layers:**

  * First FC: 512 × 3 × 3 = 4608 inputs → 512 outputs.
  * Output FC: 512 inputs → 12 outputs (num\_classes).

**Input Processing**

* Input: 224 × 224 RGB images.
* After six MaxPool2d(2,2) layers, spatial dimensions reduce as:
  
  224 → 112 → 56 → 28 → 14 → 7 → 3 (using floor division).
---

## Training

The model was trained with the following hyperparameters:

* Number of Classes: 12
* Batch Size: 8
* Learning Rate: 5e-4
* Epochs: 50
* Early Stopping Patience: 10
* Optimizer: AdamW
* Loss Function: Cross Entropy Loss

Run `main.py` to train the model for 50 epochs. The best model is saved to `saved_models/best_model.pth`.
**Note:** A trained model checkpoint (`best_model.pth`) is included in the `saved_models/` directory, allowing users to perform inference without retraining the model.
If you only want to test predictions, you can directly run:

```bash
python object-detection.py
```
---

## Evaluation

The model was evaluated using `validation-checker.py`, which:

* Achieves a validation accuracy of **89.39%**.
* Generates a confusion matrix to analyze classification performance across the 12 waste categories.
* Produces a classification report containing precision, recall, F1-score, and support for each class.

Predictions are supported for both webcam feeds and static images via `object-detection.py`.


---

## Usage Example

**Train the Model:**

```bash
python main.py
```

**Predict with Webcam or Image:**

```bash
python object-detection.py
```


The inference pipeline uses OpenCV for preprocessing and converts input images into float32 tensors before passing them to the CNN model.

The inference module supports:

- Real-time webcam prediction
- Static image classification
- Confidence score generation

**Example output:**


```
Predicted Class: Plastic
Confidence: 0.92
```

**Evaluate the Model:**

```bash
python validation-checker.py
```

Outputs:

* Validation accuracy: 89.39%
* Classification report
* Confusion matrix visualization


---

## Results

The model achieved a validation accuracy of **89.39%**. Below are the detailed precision, recall, F1-score, and support for each class as calculated on the validation set:

<img width="515" height="446" alt="Screenshot 2026-08-01 094742" src="https://github.com/user-attachments/assets/e2f51a1a-d616-4543-9f9e-36a5c9e4b4e1" />



**Visualizations:**

* Confusion Matrix: Shows classification performance across the 12 classes.
  <img width="797" height="691" alt="Screenshot 2026-08-01 094727" src="https://github.com/user-attachments/assets/17628784-dc22-4333-8324-f91b330fbe65" />


    
    
* Webcam Prediction: Real-time classification from webcam feed.
  
  <img width="949" alt="Screenshot 2025-05-26 201547" src="https://github.com/user-attachments/assets/edf863a2-f185-439c-ada8-eb8915c78142" />
* Image Prediction: Classification of static images using the trained CNN.
  
  <img width="164" alt="image" src="https://github.com/user-attachments/assets/6f7bd08c-b139-412c-b1e2-890eb618e075" />

---

## Author

**Khushi Keshari**

B.Tech Computer Science and Engineering  
National Institute of Technology Patna

GitHub: [Khushi Keshari](https://github.com/Khushi-Keshari)
