#!/bin/bash

# Script to run the Streamlit app

echo "Starting Brain Tumor Classification App..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0


