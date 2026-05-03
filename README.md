# Food Image Classification using Transfer Learning

## Project Overview

This project explores how pretrained deep learning models can be adapted to a new image classification task through fine-tuning. The goal is to classify food images into six categories using an ImageNet-pretrained EfficientNet-B0 model and evaluate how data augmentation and synthesized data affect model performance.

The project compares four training configurations:

1. **Baseline**: Original training data only  
2. **Original Data + Augmentation**: Original training images with online augmentation  
3. **Original Data + Synthesized Data**: Original images combined with generated synthetic variants  
4. **Original Data + Synthesized Data + Augmentation**: Synthetic dataset with online augmentation applied during training  

A Streamlit application is also included to demonstrate the best performing trained model on uploaded food images.

---

## Classes

The model classifies images into the following six food categories:

- Burger
- Pasta
- Pizza
- Salad
- Spaghetti
- Sushi

---

## Dataset Structure

The dataset is organized into train, validation, test, and synthetic training folders.

```text
data/
│
├── train/
│   ├── Burger/
│   ├── Pasta/
│   ├── Pizza/
│   ├── Salad/
│   ├── Spaghetti/
│   └── Sushi/
│
├── val/
│   ├── Burger/
│   ├── Pasta/
│   ├── Pizza/
│   ├── Salad/
│   ├── Spaghetti/
│   └── Sushi/
│
├── test/
│   ├── Burger/
│   ├── Pasta/
│   ├── Pizza/
│   ├── Salad/
│   ├── Spaghetti/
│   └── Sushi/
│
└── train_synthetic/
    ├── Burger/
    ├── Pasta/
    ├── Pizza/
    ├── Salad/
    ├── Spaghetti/
    └── Sushi/