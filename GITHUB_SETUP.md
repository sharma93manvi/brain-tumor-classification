# GitHub Setup Instructions

## Quick Start

### 1. Initialize Git Repository

```bash
cd brain-tumor-classification
git init
```

Or use the setup script:
```bash
bash scripts/setup_git.sh
```

### 2. Create GitHub Repository

1. Go to [GitHub](https://github.com/new)
2. Create a new repository named: `brain-tumor-classification`
3. **Do NOT** initialize with README (we already have one)
4. Copy the repository URL

### 3. Connect and Push

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-classification.git

# Add all files
git add .

# Commit
git commit -m "Initial commit: Brain Tumor Classification project"

# Push to GitHub
git branch -M main
git push -u origin main
```

## Repository Settings

### Recommended Settings:

1. **Description**: "Deep learning system for automated brain tumor classification from MRI images"
2. **Topics**: `machine-learning`, `deep-learning`, `medical-imaging`, `pytorch`, `brain-tumor`, `mri`, `computer-vision`
3. **License**: MIT (already included)
4. **Visibility**: Public (or Private, your choice)

## What Gets Pushed

**Included:**
- Source code (`src/`)
- Notebooks (`notebooks/`)
- Documentation (`docs/`, `README.md`)
- Configuration files (`.gitignore`, `requirements.txt`)
- License and contributing guidelines

**Excluded** (via `.gitignore`):
- Model files (`models/*.pkl`, `models/*.pth`)
- Dataset (`data/`)
- Python cache (`__pycache__/`)
- IDE files (`.vscode/`, `.idea/`)

## After Pushing

Your repository will be ready with:
- Professional README
- Complete documentation
- Clean project structure
- CI/CD setup (GitHub Actions)

## Updating the Repository

```bash
# After making changes
git add .
git commit -m "Description of changes"
git push
```

## Adding Models (Optional)

If you want to share trained models:
1. Remove `models/*.pkl` and `models/*.pth` from `.gitignore`
2. Add models to git: `git add models/`
3. Commit and push

**Note**: Model files can be large. Consider using Git LFS for large files.

