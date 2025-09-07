#!/bin/bash

# run_dataset_size_experiments.sh
# Script to run dataset size experiments for all many-shot models

set -e  # Exit on any error

echo "=== Running Dataset Size Experiments ==="

# Configuration
DATASET_SIZES=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150)
MODELS=("yolov5n" "yolov5s" "yolov5m" "yolov5l" "yolov5x" "yolov8n" "yolov8s" "yolov8m" "yolov8l" "yolov8x" "yolo11n" "yolo11s" "yolo11m" "yolo11l" "yolo11x" "rtdetr-l" "rtdetr-x")
DATASETS=("1" "2" "3")  # ID datasets
EXPERIMENT_TYPE="dataset_size"

# Create results directory
mkdir -p experiments/results/dataset_size
mkdir -p logs/dataset_size

# Function to train model with specific dataset size
train_model() {
    local model=$1
    local dataset_id=$2
    local size=$3
    local data_source=$4  # "ID" or "OOD"
    
    local exp_name="${model}_dataset_${dataset_id}_${data_source}_size_${size}"
    local log_file="logs/dataset_size/${exp_name}.log"
    
    echo "Training $exp_name..."
    
    # Prepare dataset path
    if [ "$data_source" == "ID" ]; then
        local dataset_path="dataset/${dataset_id}/yolo_datasets/DatasetSize_${size}_Train_$((size*9/10))_Val_$((size/10))/dataset.yaml"
    else
        local dataset_path="dataset/AllDatasets/dataset.yaml"
    fi
    
    # Training command
    yolo detect train \
        model=${model}.pt \
        data="$dataset_path" \
        epochs=300 \
        imgsz=224 \
        batch=16 \
        patience=50 \
        project="experiments/models/dataset_size" \
        name="$exp_name" \
        exist_ok=True \
        verbose=True > "$log_file" 2>&1
    
    echo "Completed training $exp_name"
}

# Function to evaluate model
evaluate_model() {
    local model_path=$1
    local test_dataset=$2
    local exp_name=$3
    
    echo "Evaluating $exp_name..."
    
    # Validation
    yolo detect val \
        model="$model_path" \
        data="$test_dataset" \
        imgsz=224 \
        batch=16 \
        project="experiments/results/dataset_size" \
        name="${exp_name}_val" \
        exist_ok=True
    
    # Prediction with SAHI
    python scripts/predict_with_sahi.py \
        --model "$model_path" \
        --source "dataset/test_images" \
        --output "experiments/predictions/dataset_size/${exp_name}" \
        --conf-thresholds "0,0.05,0.1,0.15,0.2,0.25,0.29,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99"
}

# Main experiment loop
echo "Starting dataset size experiments..."

# In-Domain experiments
echo "Running In-Domain (ID) experiments..."
for dataset_id in "${DATASETS[@]}"; do
    echo "Processing dataset $dataset_id..."
    
    for size in "${DATASET_SIZES[@]}"; do
        echo "Dataset size: $size images"
        
        # Prepare dataset split for this size
        python scripts/prepare_dataset_split.py \
            --dataset-id "$dataset_id" \
            --size "$size" \
            --output-dir "dataset/${dataset_id}/yolo_datasets/DatasetSize_${size}_Train_$((size*9/10))_Val_$((size/10))"
        
        for model in "${MODELS[@]}"; do
            # Generate unique experiment ID
            exp_id=$((RANDOM % 1000 + 1))
            
            # Train model
            train_model "$model" "$dataset_id" "$size" "ID"
            
            # Evaluate model
            model_path="experiments/models/dataset_size/${model}_dataset_${dataset_id}_ID_size_${size}/weights/best.pt"
            test_dataset="dataset/${dataset_id}/yolo_datasets/test/dataset.yaml"
            
            if [ -f "$model_path" ]; then
                evaluate_model "$model_path" "$test_dataset" "${model}_dataset_${dataset_id}_ID_size_${size}"
            else
                echo "Warning: Model not found at $model_path"
            fi
        done
    done
done

# Out-of-Distribution experiments
echo "Running Out-of-Distribution (OOD) experiments..."

# Prepare combined OOD dataset
python scripts/prepare_ood_dataset.py --output-dir "dataset/AllDatasets"

for dataset_id in "${DATASETS[@]}"; do
    echo "Testing OOD models on dataset $dataset_id..."
    
    for size in "${DATASET_SIZES[@]}"; do
        echo "OOD dataset size: $size images"
        
        # Use subset of OOD data
        python scripts/prepare_ood_subset.py \
            --size "$size" \
            --output-dir "dataset/AllDatasets/subset_${size}"
        
        for model in "${MODELS[@]}"; do
            # Generate unique experiment ID
            exp_id=$((RANDOM % 1000 + 1))
            
            # Train on OOD data
            train_model "$model" "$dataset_id" "$size" "OOD"
            
            # Evaluate on ID test data
            model_path="experiments/models/dataset_size/${model}_dataset_${dataset_id}_OOD_size_${size}/weights/best.pt"
            test_dataset="dataset/${dataset_id}/yolo_datasets/test/dataset.yaml"
            
            if [ -f "$model_path" ]; then
                evaluate_model "$model_path" "$test_dataset" "${model}_dataset_${dataset_id}_OOD_size_${size}"
            else
                echo "Warning: Model not found at $model_path"
            fi
        done
    done
done

# Generate analysis results
echo "Generating analysis results..."
python scripts/analyze_dataset_size_results.py \
    --results-dir "experiments/results/dataset_size" \
    --output-dir "experiments/analysis/dataset_size"

# Create performance plots
echo "Creating performance plots..."
python plot_many_shot_size_result.py

echo "=== Dataset Size Experiments Completed ==="
echo ""
echo "Results saved in:"
echo "  - experiments/results/dataset_size/"
echo "  - experiments/analysis/dataset_size/"
echo ""
echo "Next steps:"
echo "  1. Review the generated plots and analysis"
echo "  2. Run dataset quality experiments with: bash scripts/run_dataset_quality_experiments.sh"
echo "  3. Fit empirical models to determine minimum dataset requirements"
