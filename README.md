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
- **Web Application**: Interactive Streamlit app for easy model inference

## Performance

| Model | Approach | Accuracy | F1-Score (Macro) |
|-------|----------|----------|------------------|
| Baseline | Manual Features | 68.59% | 0.69 |
| ResNet50 | Feature Extraction | 91.34% | 0.91 |
| EfficientNet-B3 | Feature Extraction | **91.34%** | **0.91** |
| EfficientNet-B3 | Fine-Tuning | **97.20%** | **0.97** |

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Hardware Requirements

**Training Environment:**
- **GPU**: NVIDIA A100 GPU (used for model training)
- **Note**: Training can be performed on CPU, but GPU acceleration significantly reduces training time, especially for fine-tuning the EfficientNet-B3 model

**Inference:**
- CPU is sufficient for inference using the Streamlit web application

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd brain-tumor-classification

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Web Application (Recommended)

Run the Streamlit web app for interactive predictions:

```bash
streamlit run app.py
```

Or use the provided script:
```bash
./scripts/run_app.sh
```

Then open your browser to `http://localhost:8501`

#### Option 2: Jupyter Notebook

1. **Data Preparation**: Place your MRI images in the `data/` directory
2. **Run Notebook**: Open `notebooks/brain_tumor_classification.ipynb`
3. **Train Models**: Follow the notebook cells to train and evaluate models
4. **Use Trained Models**: Load saved models from `models/` directory

## Project Structure

```
brain-tumor-classification/
├── app.py             # Streamlit web application
├── notebooks/         # Jupyter notebooks for analysis
├── models/            # Saved model files
├── data/              # Dataset (not included in repo)
├── src/               # Source code modules
│   ├── preprocessing.py      # Medical image preprocessing
│   └── feature_extractor.py  # Feature extraction classes
├── docs/              # Documentation
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   └── STREAMLIT_APP.md
├── scripts/           # Utility scripts
│   ├── setup_git.sh
│   └── run_app.sh
├── .streamlit/        # Streamlit configuration
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

The EfficientNet-B3 fine-tuned model achieves:
- **97.20% accuracy** on test set
- **+28.61% improvement** over baseline manual features (68.59%)
- **+5.86% improvement** over feature extraction approach (91.34%)
- **Excellent performance** across all tumor classes

## Technologies

- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **OpenCV**: Image preprocessing
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Streamlit**: Web application framework

## Web Application

The project includes a Streamlit web application (`app.py`) that provides:

- **Interactive Interface**: Upload MRI images and get instant predictions
- **Multiple Models**: Choose between fine-tuned or feature extraction approaches
- **Visualization**: See preprocessing steps and probability distributions
- **User-Friendly**: Clean, intuitive design for non-technical users

For detailed information, see [Streamlit App Documentation](docs/STREAMLIT_APP.md)

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

