from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import rasterio
from rasterio.mask import mask
import numpy as np
import fiona
import yaml
from os.path import join, exists, isfile
from os import makedirs, listdir

def main(experiment):

    # Define the paths
    pred_dir = f'experiments/predictions/{experiment}'
    exp_dir = f'experiments/models/{experiment}'
    config_dir = 'src/object_detection/config/experiment_config/'
    # Create log directory
    makedirs(pred_dir, exists_ok=True)
    log_path = join(pred_dir, 'log.txt')
    models_id = [f for f in listdir(exp_dir) if not isfile(join(exp_dir, f))]

    for model_id in models_id:
        config_id = f'config_{model_id}.yaml'
        config_path = join(config_dir, config_id)
        # Check config file exists
        if not exists(config_path):
            with open(log_path, 'a') as log_file:
                log_file.write(f"Config file {config_id} not found\n")
            continue
        # Read the YAML file
        with open(config_path, 'r') as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
        # Load the model
        model_path = yaml_content['train']['model_checkpoint']
        # Load image
        dataset_path = yaml_content['data']['path_to_dataset']
        field_shape_path = join(dataset_path, yaml_content['data']['dataset'], 'field_shape.shp')
        # Load the field shape
        with fiona.open(field_shape_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        # Load the raster
        raster_path = join(dataset_path, yaml_content['data']['dataset'], 'orthomosaic.tif')
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta.copy()
        # Update the metadata to reflect the new dimensions
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
        # Convert the cropped image to a NumPy array
        image = np.moveaxis(out_image, 0, -1)  # Move the channel axis to the last position
                    
