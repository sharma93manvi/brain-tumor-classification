# Brain Tumor Classification from MRI Scans

A deep learning system for automated brain tumor classification from MRI images using transfer learning and medical image preprocessing.

## Overview

This project implements a clinical decision-support tool that classifies brain tumors from MRI scans into four categories:
- **No Tumor**
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**

The system leverages pretrained deep learning models (ResNet50 and EfficientNet-B3) with domain-specific medical preprocessing to achieve high classification accuracy.

## Key Features

- **Medical Image Preprocessing**: Brain contour cropping and CLAHE enhancement
- **Transfer Learning**: Feature extraction using pretrained ImageNet models
- **Multiple Architectures**: ResNet50 and EfficientNet-B3 implementations
- **Fine-Tuning Support**: End-to-end training with epochs
- **Robust Evaluation**: Cross-validation and comprehensive metrics

## Performance

| Model | Approach | Accuracy | F1-Score (Macro) |
|-------|----------|----------|------------------|
| Baseline | Manual Features | 70.25% | 0.70 |
| ResNet50 | Feature Extraction | 90.90% | 0.91 |
| EfficientNet-B3 | Feature Extraction | **91.60%** | **0.91** |
| EfficientNet-B3 | Fine-Tuning | (See results) | (See results) |

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd brain-tumor-classification

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Data Preparation**: Place your MRI images in the `data/` directory
2. **Run Notebook**: Open `notebooks/brain_tumor_classification.ipynb`
3. **Train Models**: Follow the notebook cells to train and evaluate models
4. **Use Trained Models**: Load saved models from `models/` directory

## Project Structure

```
brain-tumor-classification/
├── notebooks/          # Jupyter notebooks for analysis
├── models/            # Saved model files
├── data/              # Dataset (not included in repo)
├── src/               # Source code modules
├── docs/              # Documentation
├── scripts/           # Utility scripts
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Methodology

### Preprocessing Pipeline

1. **Brain Contour Cropping**: Removes background noise and focuses on brain region
2. **CLAHE Enhancement**: Improves local contrast for better tumor visibility
3. **Standardization**: Resize and normalize for model input

### Model Architecture

- **Feature Extraction**: Frozen pretrained encoders (ResNet50/EfficientNet-B3)
- **Classifier**: Logistic Regression on extracted features
- **Fine-Tuning**: End-to-end training with all layers trainable

## Results

The EfficientNet-B3 feature extraction approach achieves:
- **91.60% accuracy** on test set
- **+21.35% improvement** over baseline manual features
- **Excellent performance** across all tumor classes

## Technologies

- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **OpenCV**: Image preprocessing
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

## Citation

If you use this project in your research, please cite:

```bibtex
@software{brain_tumor_classification,
  title = {Brain Tumor Classification from MRI Scans},
  year = {2024},
  url = {<repository-url>}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a research project. For clinical use, additional validation and regulatory approval may be required.

