import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
import argparse
from os.path import join, exists

def main(experiment):
    # Define the paths
    pred_dir = f'experiments/predictions/{experiment}'
    exp_dir = f'experiments/models/{experiment}'
    config_dir = 'src/object_detection/config/experiment_config/'
    pred_log = join(pred_dir, 'log.txt')
    # Create log directory
    if not exists(pred_dir):
        with open(pred_log, 'w') as log_file:
            log_file.write("Postprocessing predictions\n")
    # Get experiment csv
    exp_csv = join(exp_dir, 'experiment_config.csv')
    # Load the experiment csv
    exp_df = pd.read_csv(exp_csv)
    for index, row in exp_df.iterrows():
        # Get the model id
        model_id = f'{row["model"]}_dataset_{row['dataset']}_exp_{experiment}_{row['experiment']}'
        # Get the prediction path
        pred_path = join(pred_dir, f'{model_id}.shp')
        # Check if the prediction already exists
        if not exists(pred_path):
            with open(pred_log, 'a') as log_file:
                log_file.write(f"{index}-postproc: Prediction of {model_id} not found in {pred_dir} as {pred_path}\n")
            continue
        # Get the model config path
        config_id = f'config_{model_id}.yaml'
        config_path = join(config_dir, config_id)
        # Check config file exists
        if not exists(config_path):
            with open(pred_log, 'a') as log_file:
                log_file.write(f"{index}-postproc: Config file of {model_id} not found in {config_dir} as {config_path}\n")
            continue
        # Load the prediction
        pred = gpd.read_file(pred_path)
        # Load the config
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # Get the min and max distance
        min_dist = config['dataset']['min_dist_on_row']
        max_dist = config['dataset']['max_dist_on_row'] 
        # Calculate the mean box area
        mean_box_area = ((min_dist + max_dist)/ 2)**2
        # Calculate the area range
        min_area = mean_box_area - mean_box_area / 2
        max_area = mean_box_area * 4
        # Calculate the area of the boxes
        pred['area'] = pred.area
        # Select the boxes that are within the area range
        pred = pred[(pred['area'] >= min_area) & (pred['area'] <= max_area)]
        # Load tiles from dataset
        tiles = join(config['data']['path_to_dataset'], config['data']['dataset'], 'tiles.shp')
        if not exists(tiles):
            with open(pred_log, 'a') as log_file:
                log_file.write(f"{index}-postproc: Tiles not found in {tiles}\n")
            continue
        # Load the handcrafted bbox
        handcrafted_bbox_path = config['data'].get('bboxes_path')
        if not handcrafted_bbox_path or not exists(handcrafted_bbox_path):
            with open(pred_log, 'a') as log_file:
                log_file.write(f"{index}-postproc: Handcrafted bbox not found in {handcrafted_bbox_path}\n")
                handcrafted_bbox_path = join(config['data']['path_to_dataset'], config['data']['dataset'], 'Handcrafted_dataset','bboxes.csv')
                log_file.write(f"{index}-postproc: Using default handcrafted bbox in {handcrafted_bbox_path}\n")
            config['data']['bboxes_path'] = handcrafted_bbox_path
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        # Load the tiles
        tiles = gpd.read_file(tiles)
        # Load the handcrafted bbox
        handcrafted_bbox = gpd.read_file(handcrafted_bbox_path)
        # Exclude tiles with tile_name in handcrafted bbox
        tiles = tiles[tiles['NAME'].isin(handcrafted_bbox['tile_name'])]
        # Check if pred or tiles GeoDataFrame is empty
        if pred.empty:
            with open(pred_log, 'a') as log_file:
                log_file.write(f"{index}-postproc: Prediction GeoDataFrame is empty for {model_id}\n")
            continue
        if tiles.empty:
            with open(pred_log, 'a') as log_file:
                log_file.write(f"{index}-postproc: Tiles GeoDataFrame is empty for {model_id}\n")
            continue
        # Exclude predictions in tiles
        pred = gpd.overlay(pred, tiles, how='difference')
        # Save the filtered predictions
        pred.to_file(pred_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess the predictions')
    parser.add_argument('-exp', '--experiment', type=str, help='The experiment id')
    args = parser.parse_args()
    main(args.experiment)
