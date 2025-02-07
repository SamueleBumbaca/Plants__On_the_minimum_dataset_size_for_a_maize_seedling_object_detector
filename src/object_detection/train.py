from object_detection.dataset.field_data_module import FieldDataModule
from object_detection.models import DETR, YOLO
import torch
import yaml
import argparse
import subprocess
from datetime import datetime
from os.path import join, dirname, abspath
# Set the default precision for matrix multiplication to float32
torch.set_float32_matmul_precision('medium')
# Main function to train the model
def main(config):
    with open(config, 'r') as file:
        cfg= yaml.safe_load(file)
    # Set the seed for reproducibility
    seed = cfg['train']['seed']
    torch.manual_seed(seed)
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get the git commit version
    try:
        git_commit_version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        git_commit_version = 'unknown'
    cfg['git_commit_version'] = git_commit_version
    # Initialize the data module
    data = FieldDataModule(config)
    cfg['data']['yolo_dataset'] = data.convert_to_yolo_format()
    # Load the model type from the config file
    model_type = cfg['train']['model']
    # Load the model class based on the model type
    if 'YOLO' in model_type.upper():
        print("Using YOLO model")
        model = YOLO.YOLO_model(cfg)
    elif 'DETR' in model_type.upper():
        print("Using DETR model")
        model = DETR.DETR_model(cfg)
    else:
        raise NotImplementedError

    model.train_model()
    print("Training completed")

# Entry point of the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with the given configuration.')
    parser.add_argument('-cfg', '--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)