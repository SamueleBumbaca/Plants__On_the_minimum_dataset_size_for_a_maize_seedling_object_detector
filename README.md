# On the Minimum Dataset Requirements for Fine-Tuning an Object Detector for Arable Crop Plant Counting: A Case Study on Maize Seedlings

This repository contains the code and experiments to replicate the study "On the Minimum Dataset Requirements for Fine-Tuning an Object Detector for Arable Crop Plant Counting: A Case Study on Maize Seedlings" published in Remote Sensing.

## Abstract

This study investigates the minimum dataset requirements for accurate maize seedling detection in georeferenced orthomosaics across different object detection paradigms, including traditional deep learning models (YOLOv5, YOLOv8, YOLO11, RT-DETR), few-shot methods (CD-ViTO), zero-shot approaches (OWLv2), and handcrafted computer vision algorithms.

## Repository Structure

```
├── dataset/                           # Dataset directories
│   ├── 1/                            # ID_1 dataset (V3 growth stage)
│   ├── 2/                            # ID_2 dataset (V3 growth stage)
│   ├── 3/                            # ID_3 dataset (V5 growth stage)
│   ├── AllDatasets/                  # Combined OOD scientific datasets
│   ├── DavidEtAl/                    # David et al. 2021 dataset
│   ├── Liu/                          # Liu et al. 2022 dataset
│   ├── NotGeo/                       # Non-georeferenced datasets
│   └── out_domain/                   # Out-of-distribution datasets
├── experiments/                       # Experiment configurations and results
│   ├── config_template.yaml         # Template configuration file
│   ├── dataset_split/               # Dataset split configurations
│   ├── models/                      # Trained model weights
│   │   ├── dataset_size/           # Models for dataset size experiments
│   │   └── dataset_quality/        # Models for dataset quality experiments
│   ├── predictions/                 # Model predictions
│   └── results/                     # Experiment results
├── src/                              # Source code
│   └── object_detection/
│       └── dataset/
│           └── handcrafted_method.py # Handcrafted algorithm implementation
├── few_shot/                         # Few-shot learning implementation
├── zero_shot/                        # Zero-shot learning implementation
├── scripts/                          # Execution scripts (created by setup)
├── *.ipynb                          # Jupyter notebooks for analysis
├── *.py                             # Python scripts for plotting and analysis
└── *.pt                             # Pre-trained model weights
```

## Requirements

### System Requirements
- **Operating System**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.10+ (required for match-case statements)
- **Hardware**: 
  - GPU with 24GB+ VRAM (NVIDIA RTX A5000 or equivalent)
  - 64GB+ RAM recommended
  - Intel Xeon E5-2670 v3 @ 2.30 GHz or equivalent

### Software Dependencies

#### Core Python Packages
```bash
# Scientific computing
numpy>=1.24.3
torch>=2.0.1
PyYAML>=6.0

# Computer Vision and Object Detection
ultralytics>=8.0.0
opencv-python>=4.6.0
scikit-image>=0.21.0
scikit-learn>=1.3.0

# Geospatial processing
rasterio>=1.3.8
shapely>=2.0.1
fiona>=1.9.4

# Visualization
matplotlib>=3.7.1

# Additional packages
transformers>=4.20.0  # For OWLv2 zero-shot detection
detectron2>=0.6      # For CD-ViTO few-shot detection
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd On_the_minimum_dataset_size_and_quality_for_a_maize_seedling_object_detector
```

### 2. Set Up Environment
```bash
# Create conda environment
conda create -n virtual_env python=3.10
conda activate virtual_env

# Install dependencies from environment.yml
conda env update -f environment.yml
```

### 3. Set Python Path
```bash
# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 4. Prepare Datasets
The repository expects datasets in the following structure:
```
dataset/
├── 1/                     # ID_1 dataset
├── 2/                     # ID_2 dataset  
├── 3/                     # ID_3 dataset
├── AllDatasets/           # Combined OOD datasets
├── DavidEtAl/            # David et al. 2021 dataset
├── LiuEtAl/              # Liu et al. 2022 dataset
└── NotGeo/               # Non-georeferenced datasets
```

Each ID dataset should contain:
- `orthomosaic.tif`: Georeferenced orthomosaic
- `tiles.shp`: Tile boundaries shapefile
- `field_shape.shp`: Field boundary shapefile
- Annotations in appropriate format

## Experiments Replication

The repository uses a systematic experiment management system based on CSV configuration files and a unified `run.py` script.

### 1. Handcrafted Algorithm Evaluation
```bash
# Run handcrafted object detector on all ID datasets
python src/object_detection/dataset/handcrafted_method.py --config experiments/config_template.yaml
```

### 2. Dataset Size Experiments (Many-shot Models)
The dataset size experiments are controlled by the configuration CSV file:
```bash
# Run all dataset size experiments defined in the CSV
python src/object_detection/run_exp/run.py --experiment dataset_size
```

**Configuration**: `experiments/models/dataset_size/experiment_config.csv` contains:
- **Models tested**: YOLOv5 (n,s,m,l,x), YOLOv8 (n,s,m,l,x), YOLO11 (n,s,m,l,x), RT-DETR (l,x)
- **Dataset sizes**: From 10 to 150 images (various train/val splits)
- **Datasets**: ID_1, ID_2, ID_3, plus OOD datasets (AllDatasets, DavidEtAl, LiuEtAl, etc.)
- **Training sources**: In-domain (ID) vs Out-of-distribution (OOD)

The CSV file defines 954 experiments with columns:
- `done`: Experiment completion flag
- `experiment`: Unique experiment ID
- `dataset`: Dataset identifier (1,2,3, or OOD dataset names)
- `train_size`: Training data proportion (0.1-0.7)
- `val_size`: Validation data proportion (0.1-0.5)
- `dataset_size`: Total dataset size (100-150 images)
- `model`: Model architecture (yolov5lu, yolov8n, rtdetr-l, etc.)
- `out_domain`: Whether using out-of-domain data

### 3. Dataset Quality Experiments
```bash
# Run dataset quality experiments defined in the CSV
python src/object_detection/run_exp/run.py --experiment dataset_quality
```

**Configuration**: `experiments/models/dataset_quality/experiment_config.csv` controls:
- Systematic reduction of annotation quality from 100% to 10%
- Fixed dataset size with varying annotation completeness
- Model robustness testing across different quality levels

### 4. Few-shot Learning Experiments
```bash
# Run few-shot experiments
python src/object_detection/run_exp/run.py --experiment fewshot_quality
```

The few-shot implementation tests:
- 1, 5, 10, 30, and 50 shots
- Different ViT backbone architectures
- Cross-domain prototype matching approaches

### 5. Zero-shot Learning Experiments
Zero-shot experiments are implemented in separate Jupyter notebooks:
```bash
# Run zero-shot experiments
jupyter notebook zero_shots.ipynb
```

This includes:
- Multiple text prompts for maize seedling detection
- Different model variants and ensemble methods
- Performance evaluation across various prompt strategies

## Analysis and Visualization

The repository includes several analysis scripts and Jupyter notebooks for examining experimental results:

### Main Analysis Scripts
- `plot_many_shot_size_result.py`: Visualizes dataset size analysis results
- `plot_many_shot_quality_result.py`: Visualizes dataset quality analysis results  
- `dataset_size_analysis.ipynb`: Interactive analysis of dataset size experiments
- `dataset_quality_analysis.ipynb`: Interactive analysis of dataset quality experiments
- `post_proc.ipynb`: Post-processing analysis workflows

### Out-of-Domain Evaluation
- `Datasets_out_domain.ipynb`: Analysis of out-of-domain dataset performance
- `zero_shots.ipynb`: Zero-shot learning experiments  
- `few_shots.ipynb`: Few-shot learning experiments

### Dataset Visualization
- `plot_yolo_dataset.ipynb`: Visualize YOLO dataset structure and annotations

### Generate Performance Plots
```bash
# Plot dataset size vs performance relationships
python plot_many_shot_size_result.py

# Plot dataset quality vs performance relationships  
python plot_many_shot_quality_result.py

# Plot YOLO dataset visualization
jupyter notebook plot_yolo_dataset.ipynb
```

### Comprehensive Analysis Notebooks
```bash
# Dataset size analysis
jupyter notebook dataset_size_analysis.ipynb

# Dataset quality analysis
jupyter notebook dataset_quality_analysis.ipynb

# Out-of-domain datasets analysis
jupyter notebook Datasets_out_domain.ipynb

# Post-processing and evaluation
jupyter notebook post_proc.ipynb
```

### Experiment Management
The `run.py` script coordinates four main phases:
1. **Training** (`mod.py`): Model training with specified configurations
2. **Prediction** (`pred.py`): Inference on test datasets
3. **Post-processing** (`postproc.py`): Results aggregation and metrics calculation
4. **Evaluation** (`eval.py`): Performance assessment and comparison

## Key Experimental Parameters

### Model Training Parameters (from config_template.yaml)
- **Image size**: 224×224 pixels
- **Ground Sampling Distance**: 5 mm/pixel  
- **Batch size**: 16
- **Max epochs**: 200
- **Patience**: 15 (early stopping)
- **Optimizer**: Auto (AdamW/SGD)
- **Workers**: 8

### Dataset Configuration
- **Tile coverage**: 1.12×1.12 m field area per tile
- **Inter-row distance**: 0.75 m (typical maize spacing)
- **Intra-row distance**: 0.12-0.18 m
- **Color thresholding**: HSV color space
- **Growth stage**: V3-V5 (3rd to 5th leaf unfolded)

### Handcrafted Algorithm Parameters
- **Binarization method**: HSV color thresholding
- **Distance tolerance inter-rows**: 0.25 m
- **Distance tolerance on row**: 0.05 m
- **Leaf area range**: 0.001-0.01 m²
- **Resolution**: 0.005 m/pixel

### Evaluation Metrics
- **Counting accuracy**: R² (coefficient of determination), RMSE
- **Detection accuracy**: mAP@0.5
- **Annotation quality**: MAPE (Mean Absolute Percentage Error)
- **Benchmark threshold**: R² = 0.85 (EPPO standard)

## Expected Results

### Key Findings
1. **Out-of-distribution training**: No model achieves acceptable performance (R² < 0.85)
2. **In-domain training**: Models reach benchmark with 60-130 annotated images
3. **Architecture differences**: 
   - Transformer-based (RT-DETR): ~60 images required
   - CNN-based (YOLO variants): 110-130 images required
4. **Annotation quality tolerance**: Models maintain performance with 65-90% annotation quality
5. **Few-shot/Zero-shot**: Do not meet minimum requirements for precision agriculture deployment

### Performance Benchmarks
- **Handcrafted algorithm**: R² > 0.85, RMSE < 0.2, mAP > 0.7
- **Best many-shot models**: R² > 0.85 with sufficient in-domain data
- **Few-shot models**: Best performance with 50 shots, still below benchmark
- **Zero-shot models**: Variable performance depending on text prompts

## Dataset Information

### In-Domain (ID) Datasets
- **ID_1**: V3 growth stage, 150 train + 20 test tiles
- **ID_2**: V3 growth stage, 150 train + 20 test tiles  
- **ID_3**: V5 growth stage, 150 train + 20 test tiles
- **Source**: Drone-captured (Phantom 4 Pro v2.0) at 10m altitude
- **GSD**: 2.7 mm/pixel (resampled to 5 mm/pixel)
- **Tile size**: 224×224 pixels (1.12×1.12 m field coverage)

### Out-of-Distribution (OOD) Datasets
- **DavidEtAl.2021**: 182 tiles, V3 stage
- **LiuEtAl.2022**: 596 tiles, V3 stage
- **Internet datasets**: 216 + 174 tiles from online repositories

## Reproducibility Notes

1. **Random Seeds**: All experiments use fixed random seeds for reproducibility
2. **Hardware Consistency**: Results may vary on different GPU architectures
3. **Dataset Splits**: Use provided split configurations in `experiments/dataset_split/`
4. **Model Versions**: Specific package versions listed in `requirements.txt`

## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@article{bumbaca2025minimum,
  title={On the Minimum Dataset Requirements for Fine-Tuning an Object Detector for Arable Crop Plant Counting: A Case Study on Maize Seedlings},
  author={Bumbaca, Samuele and Borgogno-Mondino, Enrico},
  journal={Remote Sensing},
  year={2025},
  volume={1},
  number={1},
  pages={0},
  publisher={MDPI}
}
```

## Support

For questions or issues with replication:
1. Check the troubleshooting section below
2. Open an issue on GitHub
3. Contact: samuele.bumbaca@unito.it

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure CUDA toolkit matches PyTorch version
2. **Memory Errors**: Reduce batch size or use gradient checkpointing
3. **Dependency Conflicts**: Use provided environment.yml for exact package versions
4. **Dataset Download Failures**: Check internet connection and available disk space
5. **Model Training Divergence**: Verify dataset annotations and reduce learning rate

### Performance Optimization

1. **Speed up training**: Use mixed precision and larger batch sizes if memory allows
2. **Reduce memory usage**: Use gradient accumulation instead of larger batch sizes
3. **Faster inference**: Use TensorRT optimization for deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- University of Turin, Department of Agricultural, Forest and Food Sciences
- MDPI Remote Sensing journal
- Ultralytics YOLO implementation
- Hugging Face Transformers library
- Detectron2 framework
