from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
import argparse
from os.path import join, exists
from os import makedirs

def geodataframe_to_coco(gdf, is_prediction=False):
    annotations = []
    for idx, row in gdf.iterrows():
        annotation = {
            "id": idx,
            "image_id": row['image_id'],  # Use the encoded image_id
            "category_id": 1,  # Assuming a single category
            "bbox": [row.geometry.bounds[0], row.geometry.bounds[1], row.geometry.bounds[2] - row.geometry.bounds[0], row.geometry.bounds[3] - row.geometry.bounds[1]],
            "area": row.geometry.area,
            "iscrowd": 0
        }
        if is_prediction:
            annotation["score"] = row.get('score', 1.0)  # Use the score if available, otherwise assign a default score
        annotations.append(annotation)
    return annotations

def main(experiment):
    # Define the paths
    pred_dir = f'experiments/predictions/{experiment}'
    res_dir = 'experiments/results'
    makedirs(res_dir, exist_ok=True)
    res_csv = f'experiments/results/{experiment}.csv'
    exp_dir = f'experiments/models/{experiment}'
    config_dir = 'src/object_detection/config/experiment_config/'
    score_thresh_path = join(exp_dir, 'score_threshold.yaml')
    # Load the score thresholds
    if exists(score_thresh_path):
        with open(score_thresh_path, 'r') as f:
            score_thresholds = yaml.load(f, Loader=yaml.FullLoader)
    else:
        score_thresholds = [0.29, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    # Get experiment csv
    exp_csv = join(exp_dir, 'experiment_config.csv')
    # Load the experiment csv
    exp_df = pd.read_csv(exp_csv)
    for index, row in exp_df.iterrows():
        if row['done'] != '_':
            print(f'Skipping {row["model"]}_{row["dataset"]}_exp_{experiment}_{row["experiment"]}')
            print('Model not trained')
            continue
        # Get the model id
        model_id = f'{row["model"]}_dataset_{row["dataset"]}_exp_{experiment}_{row["experiment"]}'
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
        # Get dataset path
        eval_tiles_path = join(config['data']['path_to_dataset'], f'{row["dataset"]}', 'eval_tiles.shp')
        # Get the ground truth path
        gt_path = join(config['data']['path_to_dataset'], config['data']['dataset'], 'gt.shp')
        # Load the ground truth
        gt = gpd.read_file(gt_path)
        # Load the evaluation tiles
        eval_tiles = gpd.read_file(eval_tiles_path)
        # Select the predictions within the evaluation tiles
        pred = gpd.sjoin(pred, eval_tiles, how="inner", predicate="intersects", rsuffix='_eval')
        # Select the ground truth within the evaluation tiles
        gt = gpd.sjoin(gt, eval_tiles, how="inner", predicate="intersects", rsuffix='_eval')
        
        # Create a mapping from unique NAME values to unique image_id values
        unique_names = pd.concat([pred['NAME'], gt['NAME']]).unique()
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        
        # Encode the NAME column to image_id in both pred and gt
        pred['image_id'] = pred['NAME'].map(name_to_id)
        gt['image_id'] = gt['NAME'].map(name_to_id)
        
        # Iterate over the score thresholds
        for score_thresh in score_thresholds:
            # Filter the predictions using the score threshold
            pred_filtered = pred[pred['score'] >= score_thresh]
            if len(pred_filtered) == 0:
                continue
            # Calculate count accuracy for each 'NAME'
            count_accuracies = []
            errors = []
            gt_values = [] # Reference values
            for name, group in pred_filtered.groupby('NAME'):
                gt_group = gt[gt['NAME'] == name]
                count_accuracy = 1 - abs(len(group) / len(gt_group) - 1)
                count_accuracies.append(count_accuracy)
                errors.append(len(gt_group) - len(group))
                gt_values.append(len(gt_group))
            rmse = np.sqrt(np.mean(np.square(errors)))
            avg_count_accuracy = np.mean(count_accuracies)
            sd_count_accuracy = np.std(count_accuracies)
            # Calculate R-squared
            ss_total = np.sum(np.square(np.array(gt_values) - np.mean(gt_values)))
            ss_residual = np.sum(np.square(errors))
            if ss_total == 0:
                r_squared = float('nan')  # Avoid division by zero
            else:
                r_squared = 1 - (ss_residual / ss_total)
            # Calculate MAPE
            mape = np.mean(np.abs(np.array(errors) / np.array(gt_values))) * 100
            # Convert ground truth and predictions to COCO format
            gt_annotations = geodataframe_to_coco(gt)
            pred_annotations = geodataframe_to_coco(pred_filtered, is_prediction=True)
            
            # Create COCO objects
            coco_gt = COCO()
            coco_gt.dataset = {
                "images": [{"id": name_to_id[name], "file_name": name} for name in unique_names],
                "annotations": gt_annotations,
                "categories": [{"id": 1, "name": "object"}]
            }
            coco_gt.createIndex()
            coco_pred = coco_gt.loadRes(pred_annotations)
            
            # Evaluate the predictions
            coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract the metrics
            ap_iou_50_95 = coco_eval.stats[0]  # AP@IoU=0.50:0.95
            ap_iou_50 = coco_eval.stats[1]     # AP@IoU=0.50
            ap_iou_75 = coco_eval.stats[2]     # AP@IoU=0.75
            ap_small = coco_eval.stats[3]      # AP@IoU=0.50:0.95 (small)
            ap_medium = coco_eval.stats[4]     # AP@IoU=0.50:0.95 (medium)
            ap_large = coco_eval.stats[5]      # AP@IoU=0.50:0.95 (large)
            ar_max_1 = coco_eval.stats[6]      # AR@IoU=0.50:0.95 (maxDets=1)
            ar_max_10 = coco_eval.stats[7]     # AR@IoU=0.50:0.95 (maxDets=10)
            ar_max_100 = coco_eval.stats[8]    # AR@IoU=0.50:0.95 (maxDets=100)
            ar_small = coco_eval.stats[9]      # AR@IoU=0.50:0.95 (small)
            ar_medium = coco_eval.stats[10]    # AR@IoU=0.50:0.95 (medium)
            ar_large = coco_eval.stats[11]     # AR@IoU=0.50:0.95 (large)
            
            # Save the results
            results = {
                'model_id': model_id,
                'score_threshold': score_thresh,
                'r_squared': r_squared,
                'rmse': rmse,
                'mape': mape,
                'errors': errors,
                'gt_values': gt_values,
                'count_accuracy_avg': avg_count_accuracy,
                'count_accuracy_sd': sd_count_accuracy,
                'ap_iou_50_95': ap_iou_50_95,
                'ap_iou_50': ap_iou_50,
                'ap_iou_75': ap_iou_75,
                'ap_small': ap_small,
                'ap_medium': ap_medium,
                'ap_large': ap_large,
                'ar_max_1': ar_max_1,
                'ar_max_10': ar_max_10,
                'ar_max_100': ar_max_100,
                'ar_small': ar_small,
                'ar_medium': ar_medium,
                'ar_large': ar_large
            }
            results_df = pd.DataFrame([results])
            # Write header only if the file does not exist
            write_header = not exists(res_csv)
            results_df.to_csv(res_csv, mode='a', header=write_header, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the predictions')
    parser.add_argument('-exp', '--experiment', type=str, help='The experiment id')
    args = parser.parse_args()
    main(args.experiment)