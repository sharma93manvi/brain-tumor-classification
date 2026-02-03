# Project Structure

```
brain-tumor-classification/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── data/                       # Dataset directory (not in git)
├── docs/                       # Documentation
│   ├── INSTALLATION.md         # Installation guide
│   └── USAGE.md                # Usage guide
├── models/                     # Saved model files (gitignored)
├── notebooks/                  # Jupyter notebooks
│   └── brain_tumor_classification.ipynb
├── scripts/                    # Utility scripts
├── src/                        # Source code
│   ├── __init__.py
│   ├── preprocessing.py        # Medical image preprocessing
│   └── feature_extractor.py    # Feature extraction classes
├── .gitignore                  # Git ignore rules
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
├── PROJECT_STRUCTURE.md        # This file
├── README.md                   # Main documentation
└── requirements.txt            # Python dependencies
```

## Directory Descriptions

- **notebooks/**: Main analysis and training notebooks
- **src/**: Reusable Python modules
- **models/**: Trained model files (saved after training)
- **data/**: Dataset storage (not tracked in git)
- **docs/**: Additional documentation
- **scripts/**: Utility and helper scripts
