#!/bin/bash

# install_dependencies.sh
# Script to install all required dependencies for the maize seedling detection project

set -e  # Exit on any error

echo "=== Installing Dependencies for Maize Seedling Detection Project ==="

# Update conda
echo "Updating conda..."
conda update -n base -c defaults conda -y

# Install core scientific computing packages
echo "Installing core scientific computing packages..."
conda install -c conda-forge numpy=1.24.3 -y
conda install -c conda-forge scipy matplotlib=3.7.1 -y
conda install -c pytorch torch=2.0.1 torchvision torchaudio pytorch-cuda=11.8 -y

# Install computer vision and object detection packages
echo "Installing computer vision packages..."
pip install ultralytics>=8.0.0
pip install opencv-python>=4.6.0
pip install scikit-image>=0.21.0
pip install scikit-learn>=1.3.0

# Install geospatial processing packages
echo "Installing geospatial packages..."
conda install -c conda-forge rasterio=1.3.8 -y
conda install -c conda-forge shapely=2.0.1 -y  
conda install -c conda-forge fiona=1.9.4 -y
conda install -c conda-forge geopandas -y

# Install additional required packages
echo "Installing additional packages..."
pip install PyYAML>=6.0
pip install transformers>=4.20.0
pip install datasets>=2.0.0
pip install timm>=0.6.0
pip install wandb  # For experiment tracking

# Install Detectron2 for few-shot learning
echo "Installing Detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Jupyter and analysis packages
echo "Installing Jupyter and analysis packages..."
conda install -c conda-forge jupyter notebook ipywidgets -y
pip install seaborn>=0.11.0
pip install plotly>=5.0.0
pip install tqdm>=4.60.0

# Install SAHI for inference
echo "Installing SAHI..."
pip install sahi>=0.11.0

# Install additional utilities
echo "Installing additional utilities..."
pip install pillow>=8.0.0
pip install imageio>=2.9.0
pip install albumentations>=1.3.0

# Create required directories
echo "Creating required directories..."
mkdir -p experiments/models/dataset_size
mkdir -p experiments/models/dataset_quality
mkdir -p experiments/results/dataset_size
mkdir -p experiments/results/dataset_quality
mkdir -p experiments/predictions
mkdir -p prediction_results
mkdir -p logs

# Set Python path
echo "Setting up Python path..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)/src\"" >> ~/.bashrc

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import rasterio; print(f'Rasterio version: {rasterio.__version__}')"
python -c "import shapely; print(f'Shapely version: {shapely.__version__}')"

echo "=== Dependencies installation completed successfully ==="
echo "Please restart your terminal or run 'source ~/.bashrc' to update PATH"
