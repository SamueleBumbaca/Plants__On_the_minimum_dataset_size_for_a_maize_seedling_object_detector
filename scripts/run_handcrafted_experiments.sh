#!/bin/bash

# run_handcrafted_experiments.sh
# Script to run handcrafted algorithm experiments on all datasets

set -e  # Exit on any error

echo "=== Running Handcrafted Algorithm Experiments ==="

# Configuration
DATASETS=("1" "2" "3")  # ID datasets
EXPERIMENT_TYPE="handcrafted"

# Create results directory
mkdir -p experiments/results/handcrafted
mkdir -p logs/handcrafted

# Function to run handcrafted algorithm on a dataset
run_handcrafted_algorithm() {
    local dataset_id=$1
    local orthomosaic_path=$2
    local output_dir=$3
    
    local exp_name="handcrafted_dataset_${dataset_id}"
    local log_file="logs/handcrafted/${exp_name}.log"
    
    echo "Running handcrafted algorithm on dataset $dataset_id..."
    echo "Orthomosaic: $orthomosaic_path"
    echo "Output: $output_dir"
    
    python src/object_detection/dataset/handcrafted_method.py \
        --dataset-id "$dataset_id" \
        --orthomosaic "$orthomosaic_path" \
        --output-dir "$output_dir" \
        --config "experiments/configs/handcrafted_${dataset_id}.yaml" \
        --tile-size 224 \
        --gsd 0.005 > "$log_file" 2>&1
    
    echo "Completed handcrafted algorithm for dataset $dataset_id"
}

# Function to evaluate handcrafted results
evaluate_handcrafted_results() {
    local dataset_id=$1
    local predictions_dir=$2
    local ground_truth_dir=$3
    
    echo "Evaluating handcrafted results for dataset $dataset_id..."
    
    python scripts/evaluate_handcrafted_results.py \
        --dataset-id "$dataset_id" \
        --predictions-dir "$predictions_dir" \
        --ground-truth-dir "$ground_truth_dir" \
        --output "experiments/results/handcrafted/dataset_${dataset_id}_metrics.json"
}

# Create handcrafted algorithm configuration files
echo "Creating handcrafted algorithm configurations..."

for dataset_id in "${DATASETS[@]}"; do
    cat > "experiments/configs/handcrafted_${dataset_id}.yaml" << EOF
# Handcrafted algorithm configuration for dataset ${dataset_id}
experiment:
  id: "handcrafted_dataset_${dataset_id}"
  
data:
  path_to_dataset: "dataset"
  dataset: "${dataset_id}"
  
algorithm:
  # Color thresholding parameters (HSV)
  color_threshold:
    hue_min: 40      # Green hue range
    hue_max: 80
    saturation_min: 50
    saturation_max: 255
    value_min: 50
    value_max: 255
  
  # Plant size constraints (in pixels at 5mm/pixel)
  leaf_area_range:
    min_area: 100    # ~2.5 cm² minimum plant area
    max_area: 2000   # ~50 cm² maximum plant area
  
  # Field geometry (in meters)
  field_geometry:
    intra_row_distance: 0.15   # 15 cm between plants in row
    inter_row_distance: 0.75   # 75 cm between rows
    row_tolerance: 0.05        # 5 cm tolerance for row alignment
  
  # Image processing parameters
  processing:
    erosion_kernel_size: 3
    dilation_kernel_size: 3
    min_contour_area: 50
    gaussian_blur_kernel: 5
  
  # RANSAC parameters for row detection
  ransac:
    max_trials: 1000
    residual_threshold: 0.05   # 5 cm tolerance
    min_samples: 3             # Minimum plants per row
  
  # Clustering parameters
  clustering:
    distance_threshold: 0.1    # 10 cm for plant grouping
    min_cluster_size: 2
EOF
done

# Create handcrafted evaluation script
echo "Creating handcrafted evaluation script..."
cat > scripts/evaluate_handcrafted_results.py << 'EOF'
#!/usr/bin/env python3

import json
import numpy as np
import argparse
from pathlib import Path
import cv2
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.features import shapes

def load_ground_truth_annotations(gt_dir, dataset_id):
    """Load ground truth annotations"""
    gt_file = Path(gt_dir) / f"dataset_{dataset_id}_annotations.json"
    if gt_file.exists():
        with open(gt_file, 'r') as f:
            return json.load(f)
    return {}

def load_handcrafted_predictions(pred_dir):
    """Load handcrafted algorithm predictions"""
    pred_file = Path(pred_dir) / "predictions.json"
    if pred_file.exists():
        with open(pred_file, 'r') as f:
            return json.load(f)
    return {}

def calculate_detection_metrics(predictions, ground_truth, distance_threshold=0.1):
    """Calculate detection metrics (precision, recall, F1)"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for tile_id, gt_points in ground_truth.items():
        if tile_id not in predictions:
            total_fn += len(gt_points)
            continue
            
        pred_points = predictions[tile_id]
        gt_matched = [False] * len(gt_points)
        
        for pred_point in pred_points:
            # Find closest ground truth point
            min_distance = float('inf')
            best_gt_idx = -1
            
            for gt_idx, gt_point in enumerate(gt_points):
                if gt_matched[gt_idx]:
                    continue
                    
                distance = np.sqrt(
                    (pred_point['x'] - gt_point['x'])**2 + 
                    (pred_point['y'] - gt_point['y'])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_gt_idx = gt_idx
            
            if min_distance <= distance_threshold and best_gt_idx >= 0:
                total_tp += 1
                gt_matched[best_gt_idx] = True
            else:
                total_fp += 1
        
        # Count unmatched ground truth points as false negatives
        total_fn += sum(1 for matched in gt_matched if not matched)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn
    }

def calculate_counting_metrics(predictions, ground_truth):
    """Calculate counting accuracy metrics"""
    pred_counts = []
    gt_counts = []
    
    for tile_id, gt_points in ground_truth.items():
        gt_count = len(gt_points)
        pred_count = len(predictions.get(tile_id, []))
        
        gt_counts.append(gt_count)
        pred_counts.append(pred_count)
    
    if len(pred_counts) == 0:
        return {"r2": 0.0, "rmse": float("inf"), "mape": float("inf")}
    
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    
    # R²
    ss_res = np.sum((gt_counts - pred_counts) ** 2)
    ss_tot = np.sum((gt_counts - np.mean(gt_counts)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # RMSE
    rmse = np.sqrt(np.mean((gt_counts - pred_counts) ** 2))
    
    # MAPE
    mape = np.mean(np.abs((gt_counts - pred_counts) / np.maximum(gt_counts, 1))) * 100
    
    return {"r2": r2, "rmse": rmse, "mape": mape}

def calculate_coverage_metrics(predictions_dir):
    """Calculate dataset coverage metrics"""
    # Count how many tiles were successfully processed
    pred_file = Path(predictions_dir) / "processing_log.json"
    if pred_file.exists():
        with open(pred_file, 'r') as f:
            log_data = json.load(f)
            
        total_tiles = log_data.get("total_tiles", 0)
        processed_tiles = log_data.get("processed_tiles", 0)
        coverage_percentage = (processed_tiles / total_tiles * 100) if total_tiles > 0 else 0
        
        return {
            "total_tiles": total_tiles,
            "processed_tiles": processed_tiles,
            "coverage_percentage": coverage_percentage
        }
    
    return {"total_tiles": 0, "processed_tiles": 0, "coverage_percentage": 0}

def main():
    parser = argparse.ArgumentParser(description="Evaluate handcrafted algorithm results")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID")
    parser.add_argument("--predictions-dir", required=True, help="Predictions directory")
    parser.add_argument("--ground-truth-dir", required=True, help="Ground truth directory")
    parser.add_argument("--output", required=True, help="Output metrics file")
    
    args = parser.parse_args()
    
    # Load data
    predictions = load_handcrafted_predictions(args.predictions_dir)
    ground_truth = load_ground_truth_annotations(args.ground_truth_dir, args.dataset_id)
    
    # Calculate metrics
    detection_metrics = calculate_detection_metrics(predictions, ground_truth)
    counting_metrics = calculate_counting_metrics(predictions, ground_truth)
    coverage_metrics = calculate_coverage_metrics(args.predictions_dir)
    
    # Combine results
    results = {
        "dataset_id": args.dataset_id,
        "algorithm": "handcrafted",
        "detection_metrics": detection_metrics,
        "counting_metrics": counting_metrics,
        "coverage_metrics": coverage_metrics,
        "benchmark_met": counting_metrics["r2"] >= 0.85,
        "num_test_tiles": len(ground_truth)
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Handcrafted algorithm evaluation completed for dataset {args.dataset_id}")
    print(f"Results saved to {output_path}")
    print(f"R² = {counting_metrics['r2']:.3f}")
    print(f"RMSE = {counting_metrics['rmse']:.3f}")
    print(f"Coverage = {coverage_metrics['coverage_percentage']:.1f}%")
    print(f"Benchmark met: {results['benchmark_met']}")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/evaluate_handcrafted_results.py

# Main experiment loop
echo "Starting handcrafted algorithm experiments..."

for dataset_id in "${DATASETS[@]}"; do
    echo "Processing dataset $dataset_id..."
    
    # Check if orthomosaic exists
    orthomosaic_path="dataset/${dataset_id}/orthomosaic.tif"
    if [ ! -f "$orthomosaic_path" ]; then
        echo "Warning: Orthomosaic not found at $orthomosaic_path"
        echo "Please ensure orthomosaics are available for handcrafted algorithm testing"
        continue
    fi
    
    # Run handcrafted algorithm
    output_dir="experiments/results/handcrafted/dataset_${dataset_id}"
    run_handcrafted_algorithm "$dataset_id" "$orthomosaic_path" "$output_dir"
    
    # Evaluate results
    if [ -d "$output_dir" ]; then
        gt_dir="dataset/${dataset_id}/ground_truth"
        evaluate_handcrafted_results "$dataset_id" "$output_dir" "$gt_dir"
    else
        echo "Warning: No results found for handcrafted algorithm on dataset $dataset_id"
    fi
    
    # Log experiment
    echo "$(date): Completed handcrafted algorithm on dataset $dataset_id" >> logs/handcrafted/experiment_summary.log
done

# Generate handcrafted analysis
echo "Generating handcrafted algorithm analysis..."
python scripts/analyze_handcrafted_results.py \
    --results-dir "experiments/results/handcrafted" \
    --output-dir "experiments/analysis/handcrafted"

# Create handcrafted performance plots
echo "Creating handcrafted performance plots..."
python scripts/plot_handcrafted_results.py \
    --analysis-dir "experiments/analysis/handcrafted" \
    --output-dir "experiments/analysis/handcrafted/plots"

echo "=== Handcrafted Algorithm Experiments Completed ==="
echo ""
echo "Results saved in:"
echo "  - experiments/results/handcrafted/"
echo "  - experiments/analysis/handcrafted/"
echo ""
echo "Key findings:"
echo "  - Dataset coverage achieved by handcrafted algorithm"
echo "  - Accuracy comparison with ground truth annotations"
echo "  - Performance baseline for deep learning approaches"
echo ""
echo "Next steps:"
echo "  1. Use handcrafted results as additional training data if performance is good"
echo "  2. Compare with deep learning approaches"
echo "  3. Generate comprehensive analysis across all methods"
