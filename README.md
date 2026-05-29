# Tea Leaf Disease Classification

A deep learning project for classifying tea leaf diseases using transfer learning with VGG16 and ResNet50 in PyTorch.

## Dataset

- **Classes:** Brown Blight, Algal Leaf, White Spot
- **Images:** ~100 per class
- **Split:** 70% Train / 15% Validation / 15% Test

## Models

| Model    | Architecture | Pretrained | Frozen Layers      |
|----------|-------------|------------|--------------------|
| VGG16    | VGG-16      | ImageNet   | All feature layers |
| ResNet50 | ResNet-50   | ImageNet   | All residual blocks|

## Results

Training curves, confusion matrices, and model comparison plots are saved in the `results/` folder after running the notebook.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Place `Data_A2.zip` in the project root
2. Open `tea_leaf_disease_classification.ipynb` in Jupyter
3. Run all cells top to bottom

## Project Structure

```
├── tea_leaf_disease_classification.ipynb
├── requirements.txt
├── results/
│   ├── sample_images.png
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   ├── model_comparison_bar.png
│   └── model_comparison.csv
└── .gitignore
```

## Assignment

CSC4093/DSC4213: Deep Learning (2024/25) — Programming Assignment 02
