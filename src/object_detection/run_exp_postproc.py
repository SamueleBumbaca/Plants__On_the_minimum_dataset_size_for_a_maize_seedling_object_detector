import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
import argparse
from os.path import join

def main(experiment):
    # Define the paths
    pred_dir = f'experiments/predictions/{experiment}'
    exp_dir = f'experiments/models/{experiment}'
    config_dir = 'src/object_detection/config/experiment_config/'
    # Get experiment csv
    exp_csv = join(exp_dir, 'experiment_config.csv')
    # Load the experiment csv
    exp_df = pd.read_csv(exp_csv)
    for index, row in exp_df.iterrows():
        # Get the model id
        model_id = f'{row["model"]}_dataset_{row['dataset']}_exp_{experiment}_{row['experiment']}'
        # Get the prediction path
        pred_path = join(pred_dir, f'{model_id}.shp')
        # Get the model config path
        config_id = f'config_{model_id}.yaml'
        config_path = join(config_dir, config_id)
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
        # Load the handcrafted bbox
        handcrafted_bbox_path = config['data'].get('bboxes_path')
        # Load the tiles
        tiles = gpd.read_file(tiles)
        # Load the handcrafted bbox
        handcrafted_bbox = gpd.read_file(handcrafted_bbox_path)
        # Exclude tiles with tile_name in handcrafted bbox
        tiles = tiles[tiles['NAME'].isin(handcrafted_bbox['tile_name'])]
        # Exclude predictions in tiles
        pred = gpd.overlay(pred, tiles, how='difference')
        # Save the filtered predictions
        pred.to_file(pred_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess the predictions')
    parser.add_argument('-exp-', '--experiment', type=str, help='The experiment id')
    args = parser.parse_args()
    main(args.experiment)
