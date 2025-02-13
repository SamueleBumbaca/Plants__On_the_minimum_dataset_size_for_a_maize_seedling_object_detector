from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import rasterio
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
import fiona
from shapely.geometry import shape as shapely_shape, Point
from shapely.geometry import mapping, box as shapely_box
from shapely.affinity import affine_transform
import pandas as pd
import yaml
import argparse
from os.path import join, exists, isfile
from os import makedirs, listdir

def main(experiment):

    # Define the paths
    pred_dir = f'experiments/predictions/{experiment}'
    exp_dir = f'experiments/models/{experiment}'
    config_dir = 'src/object_detection/config/experiment_config/'
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Create log directory
    makedirs(pred_dir, exist_ok=True)
    log_path = join(pred_dir, 'log.txt')
    # Get experiment csv
    exp_csv = join(exp_dir, 'experiment_config.csv')
    # Check experiment csv exists
    if not exists(exp_csv):
        with open(log_path, 'w') as log_file:
            log_file.write("Experiment csv not found\n")
        print("Experiment csv not found")
        return
    # Get the model ids
    models_id = [f for f in listdir(exp_dir) if not isfile(join(exp_dir, f))]
    # Load the experiment csv
    exp_df = pd.read_csv(exp_csv)
    # Iterate over each model in experiment config
    for index, row in exp_df.iterrows():
        # Check if the model is succesfully trained
        if row['done'] != '_':
            if row['done'] == 'x':
                with open(log_path, 'a') as log_file:
                    log_file.write(f"{index}: Experiment {experiment} {row['experiment']}, model trainig fail\n")
            else:
                with open(log_path, 'a') as log_file:
                    log_file.write(f"{index}: Experiment {experiment} {row['experiment']} not processed\n")
            continue
        # Get the model id
        model_id = f'{row["model"]}_dataset_{row['dataset']}_exp_{experiment}_{row['experiment']}'
        # Set the prediction path
        pred_path = join(pred_dir, f'{model_id}.shp')
        # Check if the prediction already exists
        if exists(pred_path):
            with open(log_path, 'a') as log_file:
                log_file.write(f"{index}: Prediction of {model_id} already exists in {pred_dir}\n")
            continue
        # Check if the model folder is present
        if model_id not in models_id:
            with open(log_path, 'a') as log_file:
                log_file.write(f"{index}: Model folder of {model_id} not found in experiment folder {exp_dir}\n")
            continue
        # Set the model config path
        config_id = f'config_{model_id}.yaml'
        config_path = join(config_dir, config_id)
        # Check config file exists
        if not exists(config_path):
            with open(log_path, 'a') as log_file:
                log_file.write(f"{index}: Config file {config_id} not found in {config_dir}\n")
            continue
        # Read the YAML file
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        try:
            # Set model path
            model_path = config['train']['model_checkpoint']
            if not model_path or not exists(model_path):
                with open(log_path, 'a') as log_file:
                    log_file.write(f"{index}: Model checkpoint not found in {model_path}\n")
                    config['train']['model_checkpoint'] = join(exp_dir, model_id, 'weights', 'best.pt')
                    log_file.write(f"{index}: Using default model checkpoint in {config['train']['model_checkpoint']}\n")
                with open(config_path, 'w') as yaml_file:
                    yaml.dump(config, yaml_file)
                model_path = config['train']['model_checkpoint']
            # Load the model
            detection_model = AutoDetectionModel.from_pretrained(
                model_type= 'ultralytics' if 'yolo' in config['train']['model'] or 'rtdetr' in config['train']['model'] else 'torchvision',
                model_path=model_path,
                confidence_threshold=0.3,
                device=device,
            )
            # Set the dataset path
            dataset_path = config['data']['path_to_dataset']
            # Set the raster path
            raster_path = join(dataset_path, config['data']['dataset'], 'orthomosaic.tif')
            # Set the filed shape path
            eval_tiles_shape_path = join(dataset_path, config['data']['dataset'], 'eval_tiles_buffered.shp')
            # Load the field shape
            with fiona.open(eval_tiles_shape_path, "r") as shapefile:
                shapes = [shapely_shape(feature["geometry"]) for feature in shapefile]
            # Initialize the prediction lists
            eval_boxes = []
            eval_scores = []
            # Iterate over each field shape
            for shape in shapes:
                bounds = shape.bounds
                # Load the raster
                with rasterio.open(raster_path) as src:
                    out_image, _ = mask(src, [shape], crop=True)
                    crs = src.crs
                image = reshape_as_image(out_image)
                # Exclude the alpha channel
                if image.shape[2] == 4:
                    image = image[:, :, :3]
                # Get the prediction
                result = get_sliced_prediction(
                image,
                detection_model,
                slice_height = 224,
                slice_width = 224,
                overlap_height_ratio = 0.25,
                overlap_width_ratio = 0.25,
                postprocess_type = 'NMS',
                postprocess_match_metric = "IOS",
                postprocess_match_threshold = 0.5,
                postprocess_class_agnostic = False,
                verbose = 2,
                )
                # Load the prediction
                boxes = [obj.bbox.to_xyxy() for obj in result.object_prediction_list]
                scores = [obj.score.value for obj in result.object_prediction_list]
                # Project the prediction in the crs
                minx, miny, maxx, maxy = bounds # from field shape bounds
                image_width = image.shape[1]
                image_height = image.shape[0]
                tile_width = maxx - minx
                tile_height = maxy - miny
                scale_x = tile_width / image_width
                scale_y = tile_height / image_height
                transformation_matrix = [scale_x, 0, 0, -scale_y, minx, maxy]
                geo_boxes = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    points = [(xmin, ymin), (xmax, ymax)]
                    transformed_points = []
                    for x, y in points:
                        point = Point(x, y)
                        transformed_point = affine_transform(point, transformation_matrix)
                        transformed_points.append((transformed_point.x, transformed_point.y))
                    geo_boxes.append(transformed_points)
                # Update the prediction lists
                eval_boxes.extend(geo_boxes)
                eval_scores.extend(scores)
            # Set prediction shp schema
            schema = {
                'geometry': 'Polygon',
                'properties': {'score': 'float',
                                }
                }
            # Write the prediction
            with fiona.open(pred_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as shp:
                for box, score in zip(eval_boxes, eval_scores):
                    minx, miny = box[0]
                    maxx, maxy = box[1]
                    geom = shapely_box(minx, miny, maxx, maxy)
                    shp.write({
                        'geometry': mapping(geom),
                        'properties': {'score': score,
                                        }
                    })
            # Update the config file
            config['predict'] = {'prediction_path': pred_path}
            with open(config_path, 'w') as yaml_file:
                yaml.dump(config, yaml_file)

        except Exception as e:
            with open(log_path, 'a') as log_file:
                log_file.write(f"{index}: Error in model {model_id}: {e}\n")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--experiment', type=str, help='Experiment name')
    args = parser.parse_args()
    main(args.experiment)