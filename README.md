# Brain Tumor Classification from MRI Scans

A deep learning clinical decision-support tool that classifies brain tumors from MRI images into four categories: **No Tumor**, **Glioma**, **Meningioma**, and **Pituitary Tumor**. The system uses transfer learning with domain-specific medical image preprocessing, achieving **≥95.9% recall across all tumor classes** — meaning fewer than 5% of tumors are missed by the model.

**[Live Demo →](https://ai-brain-tumor-classifier.streamlit.app/)**

> **Disclaimer**: This is a research project. It is not intended to replace clinical judgment. Any clinical deployment would require additional validation and regulatory approval. The high recall and accuracy numbers are partly a reflection of the dataset — it is well-balanced, curated, and sourced from a single collection with consistent image quality. Real-world clinical data with varied scanners, protocols, and edge cases would likely yield lower numbers.

---

## Problem Statement

Brain tumor diagnosis from MRI scans is a time-intensive process that requires expert radiologists. Misclassification or delayed diagnosis can have serious consequences for patient outcomes. The core challenges include:

- **Subtype differentiation**: Gliomas and meningiomas share overlapping visual features on MRI, making them difficult to distinguish even for experienced radiologists.
- **Clinical safety**: A false negative (telling a patient they have no tumor when they do) is the worst-case scenario in this domain.
- **Diagnostic efficiency**: High-volume hospital workflows need fast, reliable second-opinion tools to support radiologists.

This project addresses these challenges by building an automated classification system that serves as a **second reader** to assist radiologists, reduce diagnostic uncertainty, and improve workflow efficiency — particularly in settings with limited specialist availability.

---

## Dataset

**Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar

| Property | Details |
|----------|---------|
| Format | 2D single-modality MRI images |
| Total samples | 5,712 images |
| Classes | 4 (Glioma, Meningioma, Pituitary Tumor, No Tumor) |
| Class balance | Approximately balanced across all four classes |
| Image dimensions | Resized to 128×128 pixels after preprocessing |

### Class Distribution

| Class | Description |
|-------|-------------|
| No Tumor (Class 0) | Healthy brain tissue with no detectable tumor |
| Glioma (Class 1) | Tumors arising from glial cells; can range from low-grade to high-grade |
| Meningioma (Class 2) | Tumors arising from the meninges (membranes surrounding the brain) |
| Pituitary (Class 3) | Tumors of the pituitary gland at the base of the brain |

### Dataset Limitations

- No segmentation masks (image-level labels only)
- No patient-level metadata (age, sex, clinical history)
- Single imaging modality (no multi-modal fusion)
- Potential biases from varying image acquisition protocols
- Less standardized than multi-institutional datasets like BraTS

---

## Data Split

All experiments use the **same 80/20 train/test split** (random seed = 42) to ensure fair comparison across approaches.

| Split | Samples | Purpose |
|-------|---------|---------|
| Training | 4,569 (80%) | Model training and feature extraction |
| Testing | 1,143 (20%) | Final held-out evaluation |

For robust performance estimation, **5-fold stratified cross-validation** was also performed on the full dataset. Stratified folds maintain class distribution across each fold to prevent evaluation bias.

---

## Methodology

### Preprocessing Pipeline

Every MRI image goes through domain-specific preprocessing before feature extraction:

1. **Brain Contour Cropping**: Convert to grayscale → Gaussian blur → threshold → morphological refinement → extract largest contour → crop to bounding box. This removes background noise and focuses on the brain region.
2. **CLAHE Enhancement**: Apply Contrast-Limited Adaptive Histogram Equalization to the luminance channel (LAB color space) to enhance local contrast and improve tumor visibility without amplifying noise.
3. **Standardization**: Resize to 128×128 pixels and normalize pixel values for model input.

### Modeling Approaches

We implemented a progressive approach from simple baselines to advanced deep learning:

#### Stage 1: Baseline (Hand-Crafted Features)
- Extracted 6 interpretable features per image: mean pixel intensity, standard deviation, edge density, entropy, GLCM contrast, and GLCM homogeneity
- Classifier: Multinomial Logistic Regression
- Purpose: Establish a transparent baseline and verify that the dataset contains sufficient signal

#### Stage 2: Feature Extraction (Transfer Learning)
- Used pretrained ImageNet models (ResNet50, EfficientNet-B3) as frozen feature extractors
- Removed classifier heads, froze encoder weights
- Extracted high-dimensional feature vectors via global average pooling:
  - ResNet50: 2,048-dimensional features
  - EfficientNet-B3: 1,536-dimensional features
- Classifier: Multinomial Logistic Regression on extracted features
- Feature standardization using training set statistics only (no data leakage)

#### Stage 3: Fine-Tuning (End-to-End Training)
- Unfroze all layers of EfficientNet-B3 for end-to-end training
- Added a custom classification head (4 output classes)
- Training configuration:
  - **Epochs**: 10
  - **Batch size**: 32
  - **Optimizer**: Adam with weight decay (L2 = 1e-4)
  - **Learning rate**: 0.0001 (conservative to avoid catastrophic forgetting)
  - **LR schedule**: StepLR (reduce by 0.5 every 5 epochs)
  - **Loss function**: CrossEntropyLoss
  - **Hardware**: NVIDIA A100 GPU

---

## Results

### Primary Metric: Recall (Sensitivity)

In a clinical setting, a **false negative** — telling a patient they have no tumor when they do — is far more dangerous than a false positive. A false positive leads to additional testing; a false negative leads to missed or delayed treatment.

For this reason, **recall (sensitivity) is the primary metric** we optimized for and report on. High recall means fewer missed diagnoses.

### Class-Wise Recall — Best Model (Fine-Tuned EfficientNet-B3)

| Class | Recall (Sensitivity) | Clinical Interpretation |
|-------|:--------------------:|------------------------|
| No Tumor | 99.33% | Correctly identifies healthy patients |
| Glioma | 95.91% | Catches 96 out of 100 glioma cases |
| Meningioma | 97.27% | Catches 97 out of 100 meningioma cases |
| Pituitary | 97.81% | Catches 98 out of 100 pituitary tumor cases |
| **Tumor detection (binary)** | **99.76%** | **Misses fewer than 1 in 400 tumors** |

All tumor classes exceed the clinical safety target of ≥85% recall, ensuring minimal false negatives.

### Full Per-Class Metrics (Fine-Tuned EfficientNet-B3)

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| No Tumor | 0.99 | 0.99 | 0.99 | 298 |
| Glioma | 0.98 | 0.96 | 0.97 | 269 |
| Meningioma | 0.94 | 0.97 | 0.96 | 256 |
| Pituitary | 0.99 | 0.98 | 0.98 | 320 |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** | **1,143** |

### Model Comparison (All Approaches)

| Model | Approach | Test Accuracy | F1-Score (Macro) |
|-------|----------|:------------:|:----------------:|
| Baseline | Hand-crafted features + Logistic Regression | 68.59% | 0.69 |
| ResNet50 | Feature Extraction (frozen) | 91.34% | 0.91 |
| EfficientNet-B3 | Feature Extraction (frozen) | 91.34% | 0.91 |
| **EfficientNet-B3** | **Fine-Tuning (10 epochs)** | **97.64%** | **0.98** |

> **A note on the 97.64% accuracy**: This number is high partly because the dataset is well-balanced across four classes and relatively clean (curated Kaggle dataset with consistent image quality). In a real clinical setting with noisier images, varied scanners, and edge-case presentations, accuracy would likely be lower. This is why we emphasize recall over accuracy — it directly measures whether the model misses tumors, which is the clinically meaningful failure mode. The cross-validation results (90.60% ± 0.49%) on the feature extraction model give a more conservative estimate of generalization performance.

### Cross-Validation Results (ResNet50 Feature Extraction)

| Fold | Accuracy | F1-Macro |
|------|:--------:|:--------:|
| 1 | 90.03% | 0.8965 |
| 2 | 90.55% | 0.9016 |
| 3 | 90.81% | 0.9042 |
| 4 | 91.42% | 0.9109 |
| 5 | 90.19% | 0.8976 |
| **Mean ± Std** | **90.60% ± 0.49%** | **0.9022 ± 0.0052** |

The low standard deviation across folds confirms the model generalizes consistently and is not dependent on a specific train/test split.

### Clinical Safety Targets and Results

| Metric | Target | Achieved |
|--------|:------:|:--------:|
| Recall for all tumor classes | ≥ 0.85 | All ≥ 0.96 |
| Tumor detection recall (binary) | — | 99.76% |
| No tumor detection specificity | — | 99.33% |
| F1-score for Glioma/Meningioma differentiation | ≥ 0.70 | 0.97 / 0.96 |
| Overall accuracy | ≥ 80% | 97.64% |

### How We Verified Results

1. **Held-out test set**: 20% of data never seen during training, used only for final evaluation
2. **Stratified 5-fold cross-validation**: Confirmed performance consistency across different data partitions (90.60% ± 0.49%)
3. **Progressive comparison**: Each model was compared against the baseline and previous approaches using the same data split (random seed = 42)
4. **Per-class analysis**: Confusion matrices and per-class precision/recall/F1 ensure no single class is being neglected
5. **Clinical safety analysis**: Dedicated recall analysis with binary tumor-vs-no-tumor evaluation

---

## Key Findings

1. **Transfer learning dramatically outperforms hand-crafted features**: +22.75 percentage points improvement (68.59% → 91.34%)
2. **Fine-tuning provides further gains**: +6.30 percentage points over frozen feature extraction (91.34% → 97.64%)
3. **Glioma/Meningioma differentiation solved**: The baseline struggled most with these overlapping classes (F1: 0.63/0.62). Fine-tuning achieves F1: 0.97/0.96.
4. **Domain-specific preprocessing is essential**: Brain contour cropping + CLAHE bridges the gap between natural image features (ImageNet) and medical imaging.
5. **Clinical safety targets met**: All tumor classes exceed the ≥85% recall threshold.

---

## Project Structure

```
brain-tumor-classification/
├── app.py                          # Streamlit web application for inference
├── notebooks/
│   └── brain_tumor_classification.ipynb  # Full analysis, training, and evaluation
├── models/                         # Saved model files
│   ├── efficientnet_b3_finetuned.pth         # Fine-tuned model (~123 MB)
│   ├── brain_tumor_classifier_efficientnet_b3.pkl
│   ├── brain_tumor_classifier_resnet50.pkl
│   ├── scaler_efficientnet_b3.pkl
│   └── scaler_resnet50.pkl
├── src/                            # Source code modules
│   ├── preprocessing.py            # Brain contour cropping + CLAHE
│   └── feature_extractor.py        # Feature extraction classes
├── data/                           # Dataset directory (not included in repo)
├── requirements.txt                # Python dependencies
├── .streamlit/config.toml          # Streamlit configuration
└── scripts/
    └── run_app.sh                  # App launch script
```

---

## How to Run

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- (Optional) NVIDIA GPU with CUDA for training; CPU is sufficient for inference

### Installation

```bash
git clone <repository-url>
cd brain-tumor-classification
pip install -r requirements.txt
```

### Run the Web Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. Upload a brain MRI image and select a model to get predictions with confidence scores.

### Reproduce the Analysis

Open `notebooks/brain_tumor_classification.ipynb` in Jupyter or Google Colab. The notebook contains the complete pipeline from data loading through final evaluation.

---

## Technologies

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, Torchvision |
| Machine Learning | scikit-learn (Logistic Regression, cross-validation, metrics) |
| Image Processing | OpenCV (contour detection, CLAHE), Pillow |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Training Hardware | NVIDIA A100 GPU (Google Colab) |

---

## Limitations and Future Work

- **Single-institution dataset**: Performance on external datasets from different scanners/protocols needs validation
- **2D images only**: 3D volumetric analysis could capture more spatial context
- **No patient metadata**: Integrating age, sex, and clinical history could improve predictions
- **Interpretability**: Deep learning models are less interpretable than hand-crafted features; Grad-CAM or attention maps could help
- **Clinical validation**: Real-world clinical trials are required before any deployment

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

```bibtex
@software{brain_tumor_classification,
  title = {Brain Tumor Classification from MRI Scans},
  year = {2024},
  url = {<repository-url>}
}
```
