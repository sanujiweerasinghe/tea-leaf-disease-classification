# Tea Leaf Disease Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project for automated classification of tea leaf diseases using transfer learning with **VGG16** and **ResNet50** pretrained on ImageNet.

---

## Overview

Tea leaf diseases cause significant crop losses if not detected early. This project applies convolutional neural networks (CNNs) with transfer learning to classify three common tea leaf diseases from images, comparing the performance of two popular architectures.

---

## Dataset

| Class | Description |
|---|---|
| Brown Blight | Caused by *Colletotrichum camelliae* |
| Algal Leaf | Caused by *Cephaleuros parasiticus* |
| White Spot | Caused by *Pestalotiopsis theae* |

- ~100 images per class
- Split: **70% Train / 15% Validation / 15% Test**
- Augmentation: horizontal flip, rotation (±15°), colour jitter

---

## Models

### VGG16 — Transfer Learning
- Pretrained on ImageNet
- Frozen: all convolutional feature layers
- Trainable: final classifier head (3-class output)

### ResNet50 — Transfer Learning
- Pretrained on ImageNet
- Frozen: all residual blocks
- Trainable: final fully-connected layer (3-class output)

---

## Results

| Model    | Test Accuracy | Test Loss | Precision | Recall | F1-Score |
|----------|:------------:|:---------:|:---------:|:------:|:--------:|
| VGG16    | 86.21%       | 0.420     | 0.8673    | 0.8621 | 0.8620   |
| ResNet50 | **87.93%**   | **0.401** | 0.8788    | 0.8793 | 0.8787   |

📄 **[Read the full project report](docs/REPORT.md)** for detailed methodology, training curves, and analysis.

### Sample Images
![Sample Images](results/sample_images.png)

### Training Curves
![Training Curves](results/training_curves.png)

### Confusion Matrices
![Confusion Matrices](results/confusion_matrices.png)

### Model Comparison
![Model Comparison](results/model_comparison_bar.png)

---

## Project Structure

```
tea-leaf-disease-classification/
├── tea_leaf_disease_classification.ipynb   # Main notebook
├── requirements.txt                         # Dependencies
├── results/
│   ├── training_curves.png
│   ├── vgg16_training_curves.png
│   ├── resnet50_training_curves.png
│   ├── confusion_matrices.png
│   ├── model_comparison_bar.png
│   ├── model_comparison.csv
│   └── sample_images.png
├── docs/
│   └── REPORT.md                            # Full project report
└── .gitignore
```

---

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Dataset
Place `Data_A2.zip` in the project root directory.

### 3. Run the Notebook
```bash
jupyter notebook tea_leaf_disease_classification.ipynb
```
Run all cells top to bottom. The notebook will:
- Extract and split the dataset automatically
- Train VGG16 and ResNet50 models
- Generate and save all plots to `results/`

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib, seaborn, pandas

See `requirements.txt` for full list.

---

## Assignment

**CSC4093 / DSC4213: Deep Learning (2024/25)**
Programming Assignment 02 — Convolutional Networks and Transfer Learning
