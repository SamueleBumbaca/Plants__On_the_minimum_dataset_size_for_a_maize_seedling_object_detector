#!/bin/bash

# download_datasets.sh
# Script to download and prepare all datasets for the maize seedling detection project

set -e  # Exit on any error

echo "=== Downloading and Preparing Datasets ==="

# Create dataset directories
echo "Creating dataset directory structure..."
mkdir -p dataset/{1,2,3,AllDatasets,AllDatasetsAndNotGeo,DavidEtAl,Liu,LiuEtAl,LiuEtAlAndNotGeo,NotGeo,out_domain,tiles}

# Function to download from URL if file doesn't exist
download_if_not_exists() {
    local url=$1
    local output_path=$2
    local description=$3
    
    if [ ! -f "$output_path" ]; then
        echo "Downloading $description..."
        wget -O "$output_path" "$url" || curl -L -o "$output_path" "$url"
    else
        echo "$description already exists, skipping download."
    fi
}

# Download Out-of-Distribution Scientific Datasets
echo "Downloading OOD scientific datasets..."

# David et al. 2021 dataset
echo "Downloading David et al. 2021 dataset..."
# Note: Replace with actual download URLs when available
# download_if_not_exists "https://example.com/david_et_al_2021.zip" "dataset/DavidEtAl/david_2021.zip" "David et al. 2021 dataset"

# Liu et al. 2022 dataset  
echo "Downloading Liu et al. 2022 dataset..."
# download_if_not_exists "https://example.com/liu_et_al_2022.zip" "dataset/Liu/liu_2022.zip" "Liu et al. 2022 dataset"

# Download OOD Internet Datasets
echo "Downloading OOD internet datasets..."

# Maize seeding dataset from Roboflow
echo "Downloading Maize seeding dataset..."
# download_if_not_exists "https://universe.roboflow.com/datasets/maize-seeding.zip" "dataset/NotGeo/maize_seeding.zip" "Maize seeding dataset"

# Maize seedling detection dataset
echo "Downloading Maize seedling detection dataset..."
# download_if_not_exists "https://universe.roboflow.com/datasets/maize-seedling-detection.zip" "dataset/NotGeo/maize_seedling_detection.zip" "Maize seedling detection dataset"

# Extract downloaded datasets
echo "Extracting datasets..."
for zip_file in dataset/*/*.zip; do
    if [ -f "$zip_file" ]; then
        echo "Extracting $(basename $zip_file)..."
        unzip -q "$zip_file" -d "$(dirname $zip_file)"
        rm "$zip_file"  # Remove zip file after extraction
    fi
done

# In-Domain datasets preparation
echo "Preparing In-Domain datasets..."
echo "Note: In-Domain datasets (ID_1, ID_2, ID_3) should be collected using the methodology described in the paper:"
echo "  - Drone: Phantom 4 Pro v2.0"
echo "  - Altitude: ~10m above ground"
echo "  - GSD: 2.7 mm/pixel (resampled to 5 mm/pixel)"
echo "  - Growth stage: V3-V5"
echo "  - Tile size: 224x224 pixels"
echo ""
echo "Please place your collected orthomosaics in the following structure:"
echo "  dataset/1/orthomosaic.tif"
echo "  dataset/2/orthomosaic.tif" 
echo "  dataset/3/orthomosaic.tif"
echo ""
echo "And corresponding shape files:"
echo "  dataset/1/tiles.shp"
echo "  dataset/1/field_shape.shp"
echo "  (similar for datasets 2 and 3)"

# Create YOLO dataset structure for each ID dataset
echo "Creating YOLO dataset structure..."
for dataset_id in 1 2 3; do
    mkdir -p "dataset/${dataset_id}/yolo_datasets/DatasetSize_10_Train_9_Val_1/"{train,val,test}
    mkdir -p "dataset/${dataset_id}/yolo_datasets/DatasetSize_10_Train_9_Val_1/"{train,val,test}/{images,labels}
    
    # Create different dataset sizes
    for size in $(seq 10 10 150); do
        mkdir -p "dataset/${dataset_id}/yolo_datasets/DatasetSize_${size}_Train_$((size*9/10))_Val_$((size/10))/"{train,val}/{images,labels}
    done
done

# Create dataset.yaml template
echo "Creating dataset.yaml template..."
cat > dataset/dataset.yaml << EOF
# Dataset configuration for maize seedling detection
# Paths should be relative to this file

path: .  # Root directory
train: train/images  # Training images directory
val: val/images      # Validation images directory  
test: test/images    # Test images directory (optional)

# Classes
nc: 1  # Number of classes
names: ['maize_seedling']  # Class names

# Additional metadata
description: "Maize seedling detection dataset for V3-V5 growth stage"
version: "1.0"
license: "Research use only"
EOF

# Set up dataset splits configuration
echo "Creating dataset split configurations..."
mkdir -p experiments/dataset_split

# Create train/val split configuration for different dataset sizes
for size in $(seq 10 10 150); do
    val_size=$((size/10))
    train_size=$((size - val_size))
    
    cat > "experiments/dataset_split/size_${size}.yaml" << EOF
# Dataset split configuration for ${size} total images
total_size: ${size}
train_size: ${train_size}
val_size: ${val_size}
train_ratio: 0.9
val_ratio: 0.1
random_seed: 42
EOF
done

# Create quality level configurations
echo "Creating annotation quality configurations..."
for quality in $(seq 10 10 100); do
    cat > "experiments/dataset_split/quality_${quality}.yaml" << EOF
# Annotation quality configuration for ${quality}% annotations
quality_level: ${quality}
annotation_percentage: 0.$(printf "%02d" $quality)
random_seed: 42
missing_annotation_strategy: "random_removal"
EOF
done

echo "=== Dataset preparation completed ==="
echo ""
echo "Next steps:"
echo "1. Place your orthomosaics and annotations in the dataset/1, dataset/2, dataset/3 directories"
echo "2. Run the handcrafted algorithm to generate additional annotations if needed"
echo "3. Use the dataset preparation scripts to create YOLO-format datasets"
echo "4. Begin training experiments with the provided scripts"
