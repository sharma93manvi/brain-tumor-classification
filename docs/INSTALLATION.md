# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd brain-tumor-classification
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```python
import torch
import torchvision
import cv2
import sklearn

print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
```

## GPU Support (Optional)

For GPU acceleration, install CUDA-enabled PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure virtual environment is activated
2. **CUDA errors**: Check GPU compatibility and CUDA installation
3. **Memory errors**: Reduce batch size in training scripts

## Next Steps

After installation, see the main README.md for usage instructions.


