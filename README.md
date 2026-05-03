# Food Image Classification using Transfer Learning

### Group 10 - Bala Swapnika Gopi & Mounika Teppola

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

The dataset is organized into train, validation, test and synthetic training folders.

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

```

The train_synthetic folder contains the original training images plus additional synthesized images generated using same-class MixUp and CutMix-style image synthesis.

## Data Augmentation vs. Synthesized Data

This project separates augmentation and synthesized data as two different strategies.

### Data Augmentation
Data augmentation is applied online during training. The original images are not saved as new files. Instead, transformations are randomly applied while the model is training.

- Random horizontal flip
- Resize
- Normalization

### Synthesized Data

Synthesized data is generated offline and saved as new image files inside data/train_synthetic. For synthesis, this project uses same-class MixUp and CutMix-style image generation:

- MixUp-style synthesis blends two images from the same class.
- CutMix-style synthesis copies a patch from one same class image into another same class image.

Since both source images belong to the same class, the generated image keeps the same class label. This makes synthesized data different from online augmentation because it creates a larger saved training dataset.

## Training Configurations

### 1. Baseline

The baseline model was trained using only the original training images.
```
Dataset: data/train
Transform: Resize + Normalize
```

### 2. Original Data + Augmentation

The augmentation model was trained using the original training data with online augmentation.
```
Dataset: data/train
Transform: Resize + RandomHorizontalFlip + Rotation + Jitter + Normalize
```
### 3. Original Data + Synthesized Data

The synthetic-data model was trained using the expanded synthetic training folder.
```
Dataset: data/train_synthetic
Transform: Resize + Normalize
```
### 4. Original Data + Synthesized Data + Augmentation

The combined model was trained using the synthetic training folder with online augmentation.
```
Dataset: data/train_synthetic
Transform: Resize + RandomHorizontalFlip + Rotation + Jitter + Normalize
```

## Results

All four configurations were evaluated on the same held-out test set.

| Configuration | Test Accuracy |
|---|---:|
| Baseline | 86.67% |
| Original Data + Augmentation | 93.33% |
| Original Data + Synthesized Data | 93.33% |
| Original Data + Synthesized Data + Augmentation | 96.60% |

The best-performing configuration was:

**Original Data + Synthesized Data + Augmentation**

This configuration achieved the highest test accuracy of **96.60%**. The result suggests that combining offline synthesized data with online augmentation provided the most useful diversity for the model. The synthesized data increased the number of training examples, while augmentation exposed the model to additional variations during training.

The Streamlit app uses:

```text
best_model_combined.pth
```

## Evaluation and Analysis

The notebook includes:

- Training and validation loss curves for all four configurations
- Training and validation accuracy curves for all four configurations
- Test accuracy for all four configurations
- Confusion matrices for each model
- Accuracy comparison table
- Bar chart comparing test accuracy across all configurations
- Side-by-side confusion matrices comparing the baseline and best performing model
- Error analysis of misclassified examples
- Robustness testing under image perturbations

## Error Analysis

The error analysis compares the baseline model with the best-performing synthetic + augmentation model. Most errors occurred among visually similar food classes. For example, pizza can sometimes be confused with pasta or salad because these classes may share similar visual features such as cheese, sauce, vegetables, toppings and plate layouts. The synthetic + augmentation model improved test accuracy compared with the baseline, suggesting that online augmentation helped the model generalize better to unseen images.

## Robustness Testing

Robustness testing was performed on the best-performing model by applying controlled perturbations to the test images.

The perturbations included:

- Gaussian noise
- Gaussian blur
- Occlusion using a black box
- Brightness change

The goal was to evaluate how sensitive the model is to image quality changes and distribution shifts.

## Streamlit Demo Application

A Streamlit web application is included to demonstrate the trained model.

The app allows users to:

1. Upload a food image
2. View the predicted food class
3. View the prediction confidence
4. View class-wise confidence scores

The app uses the best selected model:
```
best_model_combined.pth
```
## Project Files
```text
FinalProject/
│
├── app.py
├── main.ipynb
├── requirements.txt
├── best_model.pth
├── best_model_aug.pth
├── best_model_syn.pth
├── best_model_combined.pth
├── Proposal.pptx
├── augmentation_examples/
├── synthetic_augmentation_examples/
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── train_synthetic/
└── .streamlit/
    └── config.toml
```

## Setup Instructions
1. Clone the Repository
```
git clone https://github.com/MounikaTeppola09/Food-Image-Classification-using-Transfer-Learning.git
cd Food-Image-Classification-using-Transfer-Learning
```

2. Create a Virtual Environment

For Windows:

```
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
```
pip install -r requirements.txt
```
4. Verify Required Files

Make sure the following files and folders are present:

```
main.ipynb
app.py
requirements.txt
best_model_combined.pth
data/
```

The Streamlit app uses:

```
best_model_combined.pth
```

because the selected best model is the Original Data + Synthetic + Augmentation configuration.

## How to Run the Notebook

After installing dependencies, open the notebook:

```
jupyter notebook main.ipynb
```

Run the notebook cells in order to:

1. Load the dataset
2. Train the four configurations and after every configuration evaluate test accuracy and generate confusion matrices
3. Run comparison analysis
4. Run error analysis
5. Run robustness testing

## How to Run the Streamlit App

From the project folder terminal, run:

```
streamlit run app.py
```

Then open the local browser URL shown in the terminal.

Usually:

```
http://localhost:8501
```

## Requirements

Main libraries used:

- Python
- PyTorch
- Torchvision
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn
- Pillow



## Key Takeaways

- Transfer learning with EfficientNet-B0 worked well for the food classification task.
- The baseline model performed strongly, but adding augmentation and synthesized data improved generalization.
- Original Data + Synthesized Data + Augmentation achieved the best test accuracy of 96.60%.
- Combining offline synthesized images with online augmentation gave the model the most diverse training signal.
- Synthetic data alone and augmentation alone improved over the baseline, but the combined approach performed best.
- Most misclassifications occurred between visually similar food classes.
- Robustness testing showed how the model responds to noise, blur, occlusion, and brightness changes.
