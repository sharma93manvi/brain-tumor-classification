#!/bin/bash

# Setup script for initializing git repository

echo "Setting up Brain Tumor Classification repository..."

# Initialize git repository
if [ ! -d .git ]; then
    git init
    echo "Git repository initialized"
else
    echo "Git repository already exists"
fi

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Brain Tumor Classification project

- Medical image preprocessing (brain cropping + CLAHE)
- Feature extraction using ResNet50 and EfficientNet-B3
- Fine-tuning support with epochs
- Comprehensive evaluation and comparison
- Documentation and setup guides"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Add remote: git remote add origin <your-repo-url>"
echo "3. Push: git push -u origin main"
echo ""

