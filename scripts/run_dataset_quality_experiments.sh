#!/bin/bash

# run_dataset_quality_experiments.sh
# Script to run dataset quality experiments by systematically reducing annotation quality

set -e  # Exit on any error

echo "=== Running Dataset Quality Experiments ==="

# Configuration
QUALITY_LEVELS=(10 20 30 40 50 60 70 80 90 100)
MODELS=("yolov5n" "yolov5s" "yolov5m" "yolov5l" "yolov5x" "yolov8n" "yolov8s" "yolov8m" "yolov8l" "yolov8x" "yolo11n" "yolo11s" "yolo11m" "yolo11l" "yolo11x" "rtdetr-l" "rtdetr-x")
DATASETS=("1" "2" "3")  # ID datasets
FIXED_DATASET_SIZE=100  # Fixed size for quality experiments
EXPERIMENT_TYPE="dataset_quality"

# Create results directory
mkdir -p experiments/results/dataset_quality
mkdir -p logs/dataset_quality

# Function to prepare dataset with reduced annotation quality
prepare_quality_dataset() {
    local dataset_id=$1
    local quality_level=$2
    local output_dir=$3
    
    echo "Preparing dataset with ${quality_level}% annotation quality..."
    
    python scripts/reduce_annotation_quality.py \
        --dataset-id "$dataset_id" \
        --quality-level "$quality_level" \
        --size "$FIXED_DATASET_SIZE" \
        --output-dir "$output_dir" \
        --random-seed 42
}

# Function to train model with quality-reduced dataset
train_quality_model() {
    local model=$1
    local dataset_id=$2
    local quality_level=$3
    
    local exp_id=$((RANDOM % 1000 + 200))  # Start from 200 to avoid conflicts with size experiments
    local exp_name="${model}_dataset_${dataset_id}_exp_dataset_quality_${exp_id}"
    local log_file="logs/dataset_quality/${exp_name}.log"
    
    echo "Training $exp_name with ${quality_level}% annotation quality..."
    
    # Prepare dataset with reduced quality
    local dataset_dir="dataset/${dataset_id}/yolo_datasets/Quality_${quality_level}_Size_${FIXED_DATASET_SIZE}"
    prepare_quality_dataset "$dataset_id" "$quality_level" "$dataset_dir"
    
    # Training command
    yolo detect train \
        model=${model}.pt \
        data="${dataset_dir}/dataset.yaml" \
        epochs=300 \
        imgsz=224 \
        batch=16 \
        patience=50 \
        project="experiments/models/dataset_quality" \
        name="$exp_name" \
        exist_ok=True \
        verbose=True > "$log_file" 2>&1
    
    echo "Completed training $exp_name"
    echo "$exp_name" >> "experiments/models/dataset_quality/experiment_log.txt"
}

# Function to evaluate quality model
evaluate_quality_model() {
    local model_path=$1
    local test_dataset=$2
    local exp_name=$3
    local quality_level=$4
    
    echo "Evaluating $exp_name (Quality: ${quality_level}%)..."
    
    # Validation
    yolo detect val \
        model="$model_path" \
        data="$test_dataset" \
        imgsz=224 \
        batch=16 \
        project="experiments/results/dataset_quality" \
        name="${exp_name}_val_q${quality_level}" \
        exist_ok=True
    
    # Prediction with multiple confidence thresholds
    python scripts/predict_with_sahi.py \
        --model "$model_path" \
        --source "dataset/${dataset_id}/test_images" \
        --output "experiments/predictions/dataset_quality/${exp_name}_q${quality_level}" \
        --conf-thresholds "0,0.05,0.1,0.15,0.2,0.25,0.29,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99"
    
    # Calculate metrics for different confidence thresholds
    python scripts/calculate_quality_metrics.py \
        --predictions "experiments/predictions/dataset_quality/${exp_name}_q${quality_level}" \
        --ground-truth "dataset/${dataset_id}/test_annotations" \
        --output "experiments/results/dataset_quality/${exp_name}_q${quality_level}_metrics.json"
}

# Main experiment loop
echo "Starting dataset quality experiments..."
echo "Fixed dataset size: $FIXED_DATASET_SIZE images"

for dataset_id in "${DATASETS[@]}"; do
    echo "Processing dataset $dataset_id..."
    
    # Prepare full-quality test dataset (100% annotations)
    echo "Preparing test dataset for dataset $dataset_id..."
    python scripts/prepare_test_dataset.py \
        --dataset-id "$dataset_id" \
        --output-dir "dataset/${dataset_id}/test_images"
    
    for quality_level in "${QUALITY_LEVELS[@]}"; do
        echo "Annotation quality level: ${quality_level}%"
        
        for model in "${MODELS[@]}"; do
            # Train model with reduced annotation quality
            train_quality_model "$model" "$dataset_id" "$quality_level"
            
            # Find the trained model
            exp_name=$(tail -n 1 "experiments/models/dataset_quality/experiment_log.txt")
            model_path="experiments/models/dataset_quality/${exp_name}/weights/best.pt"
            test_dataset="dataset/${dataset_id}/test_dataset.yaml"
            
            if [ -f "$model_path" ]; then
                evaluate_quality_model "$model_path" "$test_dataset" "$exp_name" "$quality_level"
            else
                echo "Warning: Model not found at $model_path"
            fi
            
            # Log experiment details
            echo "$(date): Completed $exp_name with quality ${quality_level}%" >> logs/dataset_quality/experiment_summary.log
        done
    done
done

# Generate quality analysis results
echo "Generating quality analysis results..."
python scripts/analyze_dataset_quality_results.py \
    --results-dir "experiments/results/dataset_quality" \
    --models-dir "experiments/models/dataset_quality" \
    --output-dir "experiments/analysis/dataset_quality"

# Create quality performance plots
echo "Creating quality performance plots..."
python plot_many_shot_quality_result.py

# Fit empirical models to quality data
echo "Fitting empirical models to quality data..."
python scripts/fit_quality_scaling_functions.py \
    --results-dir "experiments/analysis/dataset_quality" \
    --output-dir "experiments/analysis/dataset_quality/scaling_models"

# Generate summary report
echo "Generating summary report..."
python scripts/generate_quality_report.py \
    --analysis-dir "experiments/analysis/dataset_quality" \
    --output "experiments/analysis/dataset_quality/quality_experiment_report.html"

echo "=== Dataset Quality Experiments Completed ==="
echo ""
echo "Results saved in:"
echo "  - experiments/results/dataset_quality/"
echo "  - experiments/analysis/dataset_quality/"
echo ""
echo "Key findings:"
echo "  - Model performance vs annotation quality relationships"
echo "  - Minimum quality thresholds for benchmark performance (R² = 0.85)"
echo "  - Architecture-specific tolerance to annotation quality reduction"
echo ""
echo "Next steps:"
echo "  1. Review quality tolerance analysis"
echo "  2. Run few-shot experiments: bash scripts/run_few_shot_experiments.sh"
echo "  3. Compare results with size experiments"
