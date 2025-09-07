#!/bin/bash

# download_models.sh
# Script to download pre-trained models for the maize seedling detection project

set -e  # Exit on any error

echo "=== Downloading Pre-trained Models ==="

# Create models directory
mkdir -p models/pretrained

# Function to download model if it doesn't exist
download_model() {
    local model_name=$1
    local url=$2
    local output_path="models/pretrained/${model_name}.pt"
    
    if [ ! -f "$output_path" ]; then
        echo "Downloading $model_name..."
        wget -O "$output_path" "$url" || curl -L -o "$output_path" "$url"
        echo "$model_name downloaded successfully."
    else
        echo "$model_name already exists, skipping download."
    fi
}

# Download YOLO models
echo "Downloading YOLO models..."
download_model "yolov5nu" "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
download_model "yolov5su" "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
download_model "yolov5mu" "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt"
download_model "yolov5lu" "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt"
download_model "yolov5xu" "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt"

echo "Downloading YOLOv8 models..."
download_model "yolov8n" "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
download_model "yolov8s" "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
download_model "yolov8m" "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
download_model "yolov8l" "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt"
download_model "yolov8x" "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt"

echo "Downloading YOLO11 models..."
download_model "yolo11n" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
download_model "yolo11s" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
download_model "yolo11m" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
download_model "yolo11l" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
download_model "yolo11x" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"

echo "Downloading RT-DETR models..."
download_model "rtdetr-l" "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt"
download_model "rtdetr-x" "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.pt"

# Copy models to root directory (as they appear in the repository structure)
echo "Copying models to root directory..."
cp models/pretrained/*.pt ./

# Download few-shot learning models
echo "Setting up few-shot learning models..."
mkdir -p few_shot/models

# CD-ViTO models will be downloaded automatically by the few-shot training script
echo "CD-ViTO models will be downloaded automatically during few-shot experiments."

# Zero-shot models (OWLv2) will be downloaded automatically by transformers library
echo "OWLv2 models will be downloaded automatically during zero-shot experiments."

# Create model configuration files
echo "Creating model configuration files..."
mkdir -p experiments/configs

# Configuration for YOLOv5 models
cat > experiments/configs/yolov5.yaml << EOF
# YOLOv5 model configurations
models:
  yolov5n:
    pretrained: "yolov5nu.pt"
    parameters: 1.9M
    architecture: "CNN"
    
  yolov5s:
    pretrained: "yolov5su.pt" 
    parameters: 7.2M
    architecture: "CNN"
    
  yolov5m:
    pretrained: "yolov5mu.pt"
    parameters: 21.2M
    architecture: "CNN"
    
  yolov5l:
    pretrained: "yolov5lu.pt"
    parameters: 46.5M
    architecture: "CNN"
    
  yolov5x:
    pretrained: "yolov5xu.pt"
    parameters: 86.7M
    architecture: "CNN"

# Training parameters
batch_size: 16
epochs: 300
imgsz: 224
patience: 50
EOF

# Configuration for YOLOv8 models
cat > experiments/configs/yolov8.yaml << EOF
# YOLOv8 model configurations
models:
  yolov8n:
    pretrained: "yolov8n.pt"
    parameters: 3.2M
    architecture: "CNN"
    
  yolov8s:
    pretrained: "yolov8s.pt"
    parameters: 11.2M
    architecture: "CNN"
    
  yolov8m:
    pretrained: "yolov8m.pt"
    parameters: 25.9M
    architecture: "CNN"
    
  yolov8l:
    pretrained: "yolov8l.pt"
    parameters: 43.7M
    architecture: "CNN"
    
  yolov8x:
    pretrained: "yolov8x.pt"
    parameters: 68.2M
    architecture: "CNN"

# Training parameters
batch_size: 16
epochs: 300
imgsz: 224
patience: 50
EOF

# Configuration for YOLO11 models
cat > experiments/configs/yolo11.yaml << EOF
# YOLO11 model configurations
models:
  yolo11n:
    pretrained: "yolo11n.pt"
    parameters: 2.6M
    architecture: "CNN-Transformer"
    
  yolo11s:
    pretrained: "yolo11s.pt"
    parameters: 9.4M
    architecture: "CNN-Transformer"
    
  yolo11m:
    pretrained: "yolo11m.pt"
    parameters: 20.1M
    architecture: "CNN-Transformer"
    
  yolo11l:
    pretrained: "yolo11l.pt"
    parameters: 25.3M
    architecture: "CNN-Transformer"
    
  yolo11x:
    pretrained: "yolo11x.pt"
    parameters: 56.9M
    architecture: "CNN-Transformer"

# Training parameters
batch_size: 16
epochs: 300
imgsz: 224
patience: 50
EOF

# Configuration for RT-DETR models
cat > experiments/configs/rtdetr.yaml << EOF
# RT-DETR model configurations
models:
  rtdetr-l:
    pretrained: "rtdetr-l.pt"
    parameters: 32M
    architecture: "Transformer"
    
  rtdetr-x:
    pretrained: "rtdetr-x.pt"
    parameters: 67M
    architecture: "Transformer"

# Training parameters
batch_size: 16
epochs: 300
imgsz: 224
patience: 50
EOF

echo "=== Pre-trained models download completed ==="
echo ""
echo "Downloaded models:"
ls -la *.pt
echo ""
echo "Model configurations created in experiments/configs/"
echo "Ready to start training experiments!"
