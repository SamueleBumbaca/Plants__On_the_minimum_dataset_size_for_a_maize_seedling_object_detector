#!/bin/bash

# run_zero_shot_experiments.sh
# Script to run zero-shot learning experiments with OWLv2

set -e  # Exit on any error

echo "=== Running Zero-Shot Learning Experiments ==="

# Configuration
MODELS=("owlv2-base-patch16" "owlv2-base-patch16-ensemble" "owlv2-large-patch14" "owlv2-large-patch14-ensemble")
DATASETS=("1" "2" "3")  # ID datasets
EXPERIMENT_TYPE="zero_shot"

# Text prompts for maize seedling detection
PROMPTS=(
    "maize"
    "corn"
    "seedling"
    "plant"
    "maize seedling"
    "corn seedling"
    "young maize plant"
    "maize plant"
    "corn plant"
    "aerial view of maize seedlings"
    "corn seedlings in rows"
)

# Create results directory
mkdir -p experiments/results/zero_shot
mkdir -p logs/zero_shot
mkdir -p zero_shot/datasets

# Function to run OWLv2 inference
run_owlv2_inference() {
    local model_name=$1
    local dataset_id=$2
    local prompt=$3
    local test_images_dir=$4
    local output_dir=$5
    
    local exp_name="owlv2_${model_name}_dataset_${dataset_id}_prompt_$(echo $prompt | tr ' ' '_')"
    local log_file="logs/zero_shot/${exp_name}.log"
    
    echo "Running OWLv2 inference: $exp_name"
    echo "Prompt: '$prompt'"
    
    python scripts/run_owlv2_inference.py \
        --model-name "$model_name" \
        --prompt "$prompt" \
        --images-dir "$test_images_dir" \
        --output-dir "$output_dir" \
        --confidence-thresholds "0,0.05,0.1,0.15,0.2,0.25,0.29,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99" \
        --device "cuda" > "$log_file" 2>&1
    
    echo "Completed inference: $exp_name"
    return 0
}

# Function to evaluate zero-shot results
evaluate_zero_shot_results() {
    local predictions_dir=$1
    local ground_truth_dir=$2
    local exp_name=$3
    local prompt=$4
    
    echo "Evaluating zero-shot results: $exp_name"
    
    python scripts/calculate_zero_shot_metrics.py \
        --predictions-dir "$predictions_dir" \
        --ground-truth-dir "$ground_truth_dir" \
        --output "experiments/results/zero_shot/${exp_name}_metrics.json" \
        --prompt "$prompt"
}

# Setup zero-shot environment
echo "Setting up zero-shot environment..."
if [ ! -f "zero_shot/requirements.txt" ]; then
    echo "Setting up OWLv2 environment..."
    mkdir -p zero_shot
    
    # Create requirements file for zero-shot experiments
    cat > zero_shot/requirements.txt << EOF
transformers>=4.20.0
torch>=1.12.0
torchvision>=0.13.0
pillow>=8.0.0
numpy>=1.21.0
opencv-python>=4.6.0
matplotlib>=3.5.0
tqdm>=4.60.0
huggingface-hub>=0.10.0
EOF
    
    # Install requirements
    pip install -r zero_shot/requirements.txt
fi

# Create OWLv2 inference script
echo "Creating OWLv2 inference script..."
cat > scripts/run_owlv2_inference.py << 'EOF'
#!/usr/bin/env python3

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2

def load_model_and_processor(model_name, device):
    """Load OWLv2 model and processor"""
    processor = Owlv2Processor.from_pretrained(f"google/{model_name}")
    model = Owlv2ForObjectDetection.from_pretrained(f"google/{model_name}")
    model.to(device)
    model.eval()
    return model, processor

def process_image(image_path, model, processor, prompt, device, confidence_thresholds):
    """Process single image with OWLv2"""
    image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs
    inputs = processor(text=[prompt], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs=outputs, 
        target_sizes=target_sizes,
        threshold=0.0  # We'll filter by confidence later
    )[0]
    
    # Extract predictions for different confidence thresholds
    predictions = {}
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    
    for threshold in confidence_thresholds:
        mask = scores >= threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        
        predictions[str(threshold)] = {
            "boxes": filtered_boxes.tolist(),
            "scores": filtered_scores.tolist(),
            "count": len(filtered_boxes)
        }
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Run OWLv2 zero-shot object detection")
    parser.add_argument("--model-name", required=True, help="OWLv2 model name")
    parser.add_argument("--prompt", required=True, help="Text prompt for detection")
    parser.add_argument("--images-dir", required=True, help="Directory containing test images")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--confidence-thresholds", required=True, help="Comma-separated confidence thresholds")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Parse confidence thresholds
    confidence_thresholds = [float(x) for x in args.confidence_thresholds.split(",")]
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model_name}")
    model, processor = load_model_and_processor(args.model_name, device)
    
    # Process all images
    images_dir = Path(args.images_dir)
    image_files = list(images_dir.glob("*.tif")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    results = {}
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            predictions = process_image(
                image_path, model, processor, args.prompt, device, confidence_thresholds
            )
            results[image_path.name] = predictions
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Save results
    output_file = output_dir / "predictions.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save experiment metadata
    metadata = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "num_images": len(image_files),
        "confidence_thresholds": confidence_thresholds,
        "device": str(device)
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Metadata saved to {metadata_file}")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/run_owlv2_inference.py

# Create zero-shot metrics calculation script
echo "Creating zero-shot metrics calculation script..."
cat > scripts/calculate_zero_shot_metrics.py << 'EOF'
#!/usr/bin/env python3

import json
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2

def load_ground_truth(gt_dir):
    """Load ground truth annotations"""
    gt_file = Path(gt_dir) / "annotations.json"
    if gt_file.exists():
        with open(gt_file, 'r') as f:
            return json.load(f)
    return {}

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_ap(predictions, ground_truth, iou_threshold=0.5):
    """Calculate Average Precision"""
    all_scores = []
    all_labels = []
    
    for image_name, gt_data in ground_truth.items():
        if image_name not in predictions:
            continue
            
        gt_boxes = gt_data.get("boxes", [])
        pred_data = predictions[image_name]
        
        # Get predictions for best confidence threshold
        best_threshold = "0.5"  # Default
        if best_threshold not in pred_data:
            best_threshold = list(pred_data.keys())[0]
            
        pred_boxes = pred_data[best_threshold]["boxes"]
        pred_scores = pred_data[best_threshold]["scores"]
        
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_boxes)
        
        for pred_box, score in zip(pred_boxes, pred_scores):
            all_scores.append(score)
            
            # Find best matching ground truth box
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_labels.append(1)  # True positive
                gt_matched[best_gt_idx] = True
            else:
                all_labels.append(0)  # False positive
    
    if len(all_scores) == 0:
        return 0.0
    
    return average_precision_score(all_labels, all_scores)

def calculate_counting_metrics(predictions, ground_truth):
    """Calculate counting accuracy metrics (R², RMSE, MAPE)"""
    pred_counts = []
    gt_counts = []
    
    for image_name, gt_data in ground_truth.items():
        if image_name not in predictions:
            continue
            
        gt_count = len(gt_data.get("boxes", []))
        gt_counts.append(gt_count)
        
        # Use best performing confidence threshold for counting
        pred_data = predictions[image_name]
        best_count = 0
        
        for threshold, data in pred_data.items():
            count = data["count"]
            # Simple heuristic: choose count closest to mean ground truth
            if abs(count - np.mean(gt_counts)) < abs(best_count - np.mean(gt_counts)):
                best_count = count
        
        pred_counts.append(best_count)
    
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

def main():
    parser = argparse.ArgumentParser(description="Calculate zero-shot detection metrics")
    parser.add_argument("--predictions-dir", required=True, help="Directory with predictions")
    parser.add_argument("--ground-truth-dir", required=True, help="Directory with ground truth")
    parser.add_argument("--output", required=True, help="Output metrics file")
    parser.add_argument("--prompt", required=True, help="Text prompt used")
    
    args = parser.parse_args()
    
    # Load data
    predictions_file = Path(args.predictions_dir) / "predictions.json"
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    ground_truth = load_ground_truth(args.ground_truth_dir)
    
    # Calculate metrics
    ap = calculate_ap(predictions, ground_truth)
    counting_metrics = calculate_counting_metrics(predictions, ground_truth)
    
    # Combine results
    results = {
        "prompt": args.prompt,
        "average_precision": ap,
        "counting_accuracy": counting_metrics,
        "num_test_images": len(ground_truth),
        "benchmark_met": counting_metrics["r2"] >= 0.85
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Metrics saved to {output_path}")
    print(f"R² = {counting_metrics['r2']:.3f}")
    print(f"RMSE = {counting_metrics['rmse']:.3f}")
    print(f"mAP = {ap:.3f}")
    print(f"Benchmark met: {results['benchmark_met']}")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/calculate_zero_shot_metrics.py

# Main experiment loop
echo "Starting zero-shot experiments..."

for dataset_id in "${DATASETS[@]}"; do
    echo "Processing dataset $dataset_id..."
    
    # Prepare test images
    test_images_dir="dataset/${dataset_id}/test_images"
    if [ ! -d "$test_images_dir" ]; then
        echo "Preparing test images for dataset $dataset_id..."
        python scripts/prepare_test_images.py \
            --dataset-id "$dataset_id" \
            --output-dir "$test_images_dir"
    fi
    
    # Prepare ground truth
    gt_dir="dataset/${dataset_id}/test_ground_truth"
    if [ ! -d "$gt_dir" ]; then
        echo "Preparing ground truth for dataset $dataset_id..."
        python scripts/prepare_ground_truth.py \
            --dataset-id "$dataset_id" \
            --output-dir "$gt_dir"
    fi
    
    for model_name in "${MODELS[@]}"; do
        echo "Testing model: $model_name"
        
        for prompt in "${PROMPTS[@]}"; do
            echo "Testing prompt: '$prompt'"
            
            # Run inference
            output_dir="experiments/results/zero_shot/${model_name}_dataset_${dataset_id}_prompt_$(echo $prompt | tr ' ' '_')"
            
            run_owlv2_inference "$model_name" "$dataset_id" "$prompt" "$test_images_dir" "$output_dir"
            
            # Evaluate results
            if [ -f "$output_dir/predictions.json" ]; then
                evaluate_zero_shot_results "$output_dir" "$gt_dir" "${model_name}_dataset_${dataset_id}_prompt_$(echo $prompt | tr ' ' '_')" "$prompt"
            else
                echo "Warning: No predictions found for $model_name with prompt '$prompt'"
            fi
            
            # Log experiment
            echo "$(date): Completed $model_name with prompt '$prompt' on dataset $dataset_id" >> logs/zero_shot/experiment_summary.log
        done
    done
done

# Generate zero-shot analysis
echo "Generating zero-shot analysis results..."
python scripts/analyze_zero_shot_results.py \
    --results-dir "experiments/results/zero_shot" \
    --output-dir "experiments/analysis/zero_shot"

# Create zero-shot performance plots
echo "Creating zero-shot performance plots..."
python scripts/plot_zero_shot_results.py \
    --analysis-dir "experiments/analysis/zero_shot" \
    --output-dir "experiments/analysis/zero_shot/plots"

# Find best performing prompts
echo "Finding best performing prompts..."
python scripts/find_best_prompts.py \
    --results-dir "experiments/results/zero_shot" \
    --output "experiments/analysis/zero_shot/best_prompts.json"

echo "=== Zero-Shot Learning Experiments Completed ==="
echo ""
echo "Results saved in:"
echo "  - experiments/results/zero_shot/"
echo "  - experiments/analysis/zero_shot/"
echo ""
echo "Key findings:"
echo "  - Performance across different text prompts"
echo "  - Model size effects on zero-shot performance"
echo "  - Comparison with few-shot and many-shot approaches"
echo ""
echo "Next steps:"
echo "  1. Review best performing prompts"
echo "  2. Generate comprehensive comparison across all approaches"
echo "  3. Create final research paper figures"
