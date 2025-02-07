from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import yaml

def main(experiment):

    # Define the paths
    csv_path = f'experiments/models/{experiment}/experiment_config.csv'
    output_dir = 'experiments/results/{experiment}/'
    print(f"Output directory: {output_dir}")
    print(f"CSV path: {csv_path}")
