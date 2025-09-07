#!/bin/bash

# run_few_shot_experiments.sh
# Script to run few-shot learning experiments with CD-ViTO

set -e  # Exit on any error

echo "=== Running Few-Shot Learning Experiments ==="

# Configuration
SHOT_COUNTS=(1 5 10 30 50)
BACKBONES=("ViT-S" "ViT-B" "ViT-L")
DATASETS=("1" "2" "3")  # ID datasets
EXPERIMENT_TYPE="few_shot"

# Create results directory
mkdir -p experiments/results/few_shot
mkdir -p logs/few_shot
mkdir -p few_shot/datasets

# Function to prepare few-shot dataset
prepare_few_shot_dataset() {
    local dataset_id=$1
    local shot_count=$2
    local output_dir=$3
    
    echo "Preparing ${shot_count}-shot dataset for dataset ${dataset_id}..."
    
    python scripts/prepare_few_shot_dataset.py \
        --dataset-id "$dataset_id" \
        --shot-count "$shot_count" \
        --output-dir "$output_dir" \
        --random-seed 42 \
        --annotation-format "coco"
}

# Function to train CD-ViTO model
train_cdvito_model() {
    local backbone=$1
    local dataset_id=$2
    local shot_count=$3
    local support_dir=$4
    local query_dir=$5
    
    local exp_name="cdvito_${backbone}_dataset_${dataset_id}_${shot_count}shot"
    local log_file="logs/few_shot/${exp_name}.log"
    
    echo "Training CD-ViTO with $backbone backbone, ${shot_count} shots..."
    
    # Change to few_shot directory
    cd few_shot
    
    # Training command for CD-ViTO
    python tools/train_net.py \
        --config-file "configs/cdvito/cdvito_${backbone}_few_shot.yaml" \
        --num-gpus 1 \
        DATASETS.TRAIN "('${support_dir}',)" \
        DATASETS.TEST "('${query_dir}',)" \
        OUTPUT_DIR "../experiments/models/few_shot/${exp_name}" \
        SOLVER.MAX_ITER 1000 \
        SOLVER.STEPS "(600, 800)" \
        SOLVER.BASE_LR 0.001 \
        TEST.EVAL_PERIOD 100 \
        DATALOADER.NUM_WORKERS 4 \
        MODEL.DEVICE "cuda" > "../$log_file" 2>&1
    
    # Return to main directory
    cd ..
    
    echo "Completed training $exp_name"
}

# Function to evaluate few-shot model
evaluate_few_shot_model() {
    local model_path=$1
    local test_dataset=$2
    local exp_name=$3
    local shot_count=$4
    
    echo "Evaluating $exp_name (${shot_count} shots)..."
    
    # Change to few_shot directory
    cd few_shot
    
    # Evaluation
    python tools/train_net.py \
        --config-file "configs/cdvito/cdvito_eval.yaml" \
        --eval-only \
        MODEL.WEIGHTS "$model_path" \
        DATASETS.TEST "('${test_dataset}',)" \
        OUTPUT_DIR "../experiments/results/few_shot/${exp_name}_eval" \
        MODEL.DEVICE "cuda"
    
    # Return to main directory
    cd ..
    
    # Calculate few-shot specific metrics
    python scripts/calculate_few_shot_metrics.py \
        --results-dir "experiments/results/few_shot/${exp_name}_eval" \
        --output "experiments/results/few_shot/${exp_name}_metrics.json" \
        --shot-count "$shot_count"
}

# Setup CD-ViTO environment
echo "Setting up CD-ViTO environment..."
if [ ! -d "few_shot/detectron2" ]; then
    echo "Cloning CD-ViTO repository..."
    cd few_shot
    git clone https://github.com/fanq15/cd-vito.git .
    
    # Install CD-ViTO dependencies
    pip install -e .
    
    # Download pre-trained ViT models
    mkdir -p pretrained_models
    wget -O pretrained_models/dino_vits16_pretrain.pth "https://dl.fbaipublicfiles.com/dino/dino_vits16_pretrain/dino_vits16_pretrain.pth"
    wget -O pretrained_models/dino_vitb16_pretrain.pth "https://dl.fbaipublicfiles.com/dino/dino_vitb16_pretrain/dino_vitb16_pretrain.pth"
    wget -O pretrained_models/dino_vitl16_pretrain.pth "https://dl.fbaipublicfiles.com/dino/dino_vitl16_pretrain/dino_vitl16_pretrain.pth"
    
    cd ..
fi

# Create CD-ViTO configuration files
echo "Creating CD-ViTO configuration files..."
for backbone in "${BACKBONES[@]}"; do
    backbone_lower=$(echo "$backbone" | tr '[:upper:]' '[:lower:]' | sed 's/-//g')
    
    cat > "few_shot/configs/cdvito/cdvito_${backbone}_few_shot.yaml" << EOF
_BASE_: "../Base-RCNN-FPN.yaml"
MODEL:
  META_ARCHITECTURE: "CD_ViTO"
  BACKBONE:
    NAME: "build_vit_backbone"
    VIT:
      PATCH_SIZE: 16
      EMBED_DIM: $([ "$backbone" = "ViT-S" ] && echo "384" || [ "$backbone" = "ViT-B" ] && echo "768" || echo "1024")
      DEPTH: $([ "$backbone" = "ViT-S" ] && echo "12" || [ "$backbone" = "ViT-B" ] && echo "12" || echo "24")
      NUM_HEADS: $([ "$backbone" = "ViT-S" ] && echo "6" || [ "$backbone" = "ViT-B" ] && echo "12" || echo "16")
      MLP_RATIO: 4.0
      QKV_BIAS: True
      DROP_PATH_RATE: 0.1
      PRETRAIN_IMG_SIZE: 224
      PRETRAIN_USE_CLS_TOKEN: True
      OUT_FEATURES: ["res3", "res4", "res5"]
  WEIGHTS: "pretrained_models/dino_${backbone_lower}16_pretrain.pth"
  RPN:
    IN_FEATURES: ["res3", "res4", "res5"]
  ROI_HEADS:
    IN_FEATURES: ["res3", "res4", "res5"]
    NUM_CLASSES: 1
DATASETS:
  TRAIN: ()
  TEST: ()
SOLVER:
  IMS_PER_BATCH: 4
  BASE_LR: 0.001
  STEPS: (600, 800)
  MAX_ITER: 1000
  CHECKPOINT_PERIOD: 200
INPUT:
  MIN_SIZE_TRAIN: (224,)
  MAX_SIZE_TRAIN: 224
  MIN_SIZE_TEST: 224
  MAX_SIZE_TEST: 224
TEST:
  EVAL_PERIOD: 100
DATALOADER:
  NUM_WORKERS: 4
EOF
done

# Main experiment loop
echo "Starting few-shot experiments..."

for dataset_id in "${DATASETS[@]}"; do
    echo "Processing dataset $dataset_id..."
    
    # Prepare test dataset
    echo "Preparing test dataset for dataset $dataset_id..."
    python scripts/convert_to_coco_format.py \
        --dataset-id "$dataset_id" \
        --split "test" \
        --output "few_shot/datasets/test_dataset_${dataset_id}.json"
    
    for shot_count in "${SHOT_COUNTS[@]}"; do
        echo "Few-shot experiment: ${shot_count} shots"
        
        # Prepare support set (training data)
        support_dir="few_shot/datasets/support_${dataset_id}_${shot_count}shot"
        prepare_few_shot_dataset "$dataset_id" "$shot_count" "$support_dir"
        
        # Convert to COCO format for CD-ViTO
        python scripts/convert_to_coco_format.py \
            --dataset-dir "$support_dir" \
            --output "few_shot/datasets/support_${dataset_id}_${shot_count}shot.json"
        
        for backbone in "${BACKBONES[@]}"; do
            # Train CD-ViTO model
            support_dataset="support_${dataset_id}_${shot_count}shot"
            query_dataset="test_dataset_${dataset_id}"
            
            train_cdvito_model "$backbone" "$dataset_id" "$shot_count" "$support_dataset" "$query_dataset"
            
            # Evaluate model
            model_path="experiments/models/few_shot/cdvito_${backbone}_dataset_${dataset_id}_${shot_count}shot/model_final.pth"
            
            if [ -f "$model_path" ]; then
                evaluate_few_shot_model "$model_path" "$query_dataset" "cdvito_${backbone}_dataset_${dataset_id}_${shot_count}shot" "$shot_count"
            else
                echo "Warning: Model not found at $model_path"
            fi
            
            # Log experiment
            echo "$(date): Completed CD-ViTO $backbone with ${shot_count} shots on dataset $dataset_id" >> logs/few_shot/experiment_summary.log
        done
    done
done

# Generate few-shot analysis results
echo "Generating few-shot analysis results..."
python scripts/analyze_few_shot_results.py \
    --results-dir "experiments/results/few_shot" \
    --output-dir "experiments/analysis/few_shot"

# Create few-shot performance plots
echo "Creating few-shot performance plots..."
python scripts/plot_few_shot_results.py \
    --analysis-dir "experiments/analysis/few_shot" \
    --output-dir "experiments/analysis/few_shot/plots"

# Compare with many-shot results
echo "Comparing few-shot vs many-shot performance..."
python scripts/compare_few_shot_many_shot.py \
    --few-shot-dir "experiments/analysis/few_shot" \
    --many-shot-dir "experiments/analysis/dataset_size" \
    --output "experiments/analysis/few_shot_vs_many_shot_comparison.html"

echo "=== Few-Shot Learning Experiments Completed ==="
echo ""
echo "Results saved in:"
echo "  - experiments/results/few_shot/"
echo "  - experiments/analysis/few_shot/"
echo ""
echo "Key findings:"
echo "  - Performance vs shot count relationships"
echo "  - Comparison with many-shot approaches"
echo "  - Backbone architecture effects on few-shot learning"
echo ""
echo "Next steps:"
echo "  1. Review few-shot vs many-shot comparison"
echo "  2. Run zero-shot experiments: bash scripts/run_zero_shot_experiments.sh"
echo "  3. Generate final comparison analysis"
