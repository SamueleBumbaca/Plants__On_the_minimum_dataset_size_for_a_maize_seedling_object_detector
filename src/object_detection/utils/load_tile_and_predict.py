import rasterio as rio
import numpy as np
import shapely  
import fiona
import os
import argparse

def load_tile_and_predict(tile_path, model):
    with rio.open(tile_path) as src:
        tile = src.read()
        tile = np.moveaxis(tile, 0, -1)
        tile = np.expand_dims(tile, axis=0)
        tile = tile / 255.0
        tile = tile.astype(np.float32)
    preds = model(tile)
    return preds

def write_prtedictions_to_shp(preds, tile_path):
    with rio.open(tile_path) as src:
        profile = src.profile
    with fiona.open('predictions.shp', 'a', 'ESRI Shapefile', schema={'geometry': 'Polygon', 'properties': [('class', 'int')], 'geometry': 'Polygon'}) as dst:
        for pred in preds:
            pred = shapely.geometry.box(*pred)
            dst.write({'geometry': shapely.geometry.mapping(pred), 'properties': {'class': 1}})

def main(tiles_folder, model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    for tile in os.listdir(tiles_folder):
        tile_path = os.path.join(tiles_folder, tile)
        preds = load_tile_and_predict(tile_path, model)
        write_prtedictions_to_shp(preds, tile_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tiles_folder', type=str, help='path to the folder containing the tiles')
    parser.add_argument('model_path', type=str, help='path to the YOLO model')
    args = parser.parse_args()
    main(args.tiles_folder, args.model_path)
