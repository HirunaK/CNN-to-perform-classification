# EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification

This repository contains the complete implementation, comparative benchmarks, and evaluation pipeline for **EN3150 Assignment 03**. The project focuses on designing and training resource-constrained Convolutional Neural Networks (CNNs) for edge and low-resource sensor environments using the Kaggle **Flowers-5** dataset, downscaled to a maximum resolution of \(64 \times 64\) pixels.

**Repository URL:** [CNN-to-perform-classification](https://github.com/HirunaK/CNN-to-perform-classification)

---

## Technical Overview

1. **Dataset Preprocessing:** Downscales flower images (Daisy, Dandelion, Rose, Sunflower, and Tulip) to \(64 \times 64\) pixels using area-averaging anti-aliasing interpolation (`cv2.INTER_AREA`). The dataset is split into **70% Training**, **15% Validation**, and **15% Testing**.

2. **Model Architectures:**
   - **Model A (Standard CNN):** Uses interleaved standard 2D convolution, batch normalization, ReLU activation, and max-pooling layers.
   - **Model B (Lightweight Depthwise Separable CNN):** Replaces standard convolutions with **depthwise separable convolutions** and Global Average Pooling (GAP), keeping the total number of trainable parameters strictly **under 100,000** (approximately 18K parameters).

3. **Hardware-Aware Activations:** Uses **ReLU** nonlinearities instead of computationally expensive transcendental functions such as `Sigmoid` and `Tanh`, reducing the need for exponential operations on edge microcontrollers.

4. **Optimizer Benchmarking:** Compares the convergence performance of **Adam**, **SGD**, and **SGD with Momentum** across both custom network architectures.

5. **State-of-the-Art (SOTA) Pre-trained Fine-Tuning:** Evaluates transfer learning and fine-tuning using **MobileNetV2** and **EfficientNet-B0** on the same \(64 \times 64\) dataset split.

---

## End-to-End Simulation Guide

Follow these steps to set up the environment, download the dataset, train the models, and run predictions.

### 1. Clone the Repository

```bash
git clone https://github.com/HirunaK/CNN-to-perform-classification.git
cd CNN-to-perform-classification
```

### 2. Set Up a Virtual Environment

Create and activate a Python virtual environment (`.venv`).

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade the core build tools and install the required packages from `requirements.txt`.

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Download and Prepare the Dataset

Run the dataset loader script to automatically download the dataset, downscale images to \(64 \times 64\) pixels using anti-aliasing interpolation, and partition the data into training, validation, and testing sets.

```bash
python data/dataset_loader.py
```

This generates the following directory structure:

```text
data/
├── train/
├── val/
└── test/
```

---

## Executing Pipelines and Benchmarks

### Train Custom Models (Model A and Model B)

Train both custom models for 20 or more epochs using data augmentation, spatial dropout regularization, and checkpointing.

```bash
python notebooks/02_custom_models_training.py
```

The trained model weights are saved automatically to:

- `saved_models/model_a_weights.pth`
- `saved_models/model_b_weights.pth`

### Run Optimizer Benchmark Experiments (Section 3)

Evaluate convergence trajectories for Adam, standard SGD, and SGD with Momentum across both custom architectures.

```bash
python notebooks/03_optimizer_comparison.py
```

### Fine-Tune SOTA Architectures (Section 5)

Fine-tune pre-trained MobileNetV2 and EfficientNet-B0 models using the downscaled dataset.

```bash
python notebooks/04_sota_finetuning.py
```

---

## Running Inference and Visualization

### 1. Classify Custom Images

Run the prediction script to compare the class predictions and probability distributions of Model A and Model B.

```bash
python predict.py
```

The script supports input images of different resolutions.

### 2. Generate Gradient Ascent Class Templates

Generate class activation templates to visualize the features learned by each model through input-space gradient ascent.

```bash
python utils/visualization.py
```

---

## Comparative Results

### Comprehensive Comparison Across Custom and Fine-Tuned SOTA Networks

| Model Topology | Total Parameters | Model Size | Test Accuracy (%) | F1-Score (Macro) |
|---|---:|---:|---:|---:|
| Model A (Standard) | 1,143,045 | 4.37 MB (4,475.49 KB) | 72.12% | 0.7127 |
| Model B (Lightweight) | **41,413** | **174.30 KB** | 73.56% | 0.7032 |
| MobileNetV2 | 2,230,277 | 8.64 MB | 86.87% | 0.8666 |
| EfficientNet-B0 | 4,013,953 | 15.47 MB | 85.97% | 0.8575 |

---

## License and Academic Integrity

This project was developed for the **EN3150 User-Centered Systems / Pattern Recognition** course.

All submitted work is intended to comply with the university's academic integrity and anti-plagiarism guidelines.