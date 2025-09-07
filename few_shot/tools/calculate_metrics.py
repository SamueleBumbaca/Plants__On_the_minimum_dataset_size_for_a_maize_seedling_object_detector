import json
import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def plot_bboxes(image_path, gt_bboxes, pred_bboxes, pred_scores):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot ground truth annotations in red
    for bbox in gt_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Plot predicted annotations in viridis color scale based on score
    norm = Normalize(vmin=0, vmax=1)
    for bbox, score in zip(pred_bboxes, pred_scores):
        color = cm.viridis(norm(score))
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Score')

    plt.show()

def calculate_coco_metrics(gt_path, predictions, score_thresh):
    # Load ground truth COCO format
    coco_gt = COCO(gt_path)
    
    # Filter predictions by score threshold
    filtered_preds = [p for p in predictions if p['score'] >= score_thresh]
    
    # Create COCO prediction format
    coco_dt = coco_gt.loadRes(filtered_preds)
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract requested metrics
    metrics = {
        'ap_iou_50_95': coco_eval.stats[0],
        'ap_iou_50': coco_eval.stats[1],
        'ap_iou_75': coco_eval.stats[2],
        'ap_small': coco_eval.stats[3],
        'ap_medium': coco_eval.stats[4],
        'ap_large': coco_eval.stats[5],
        'ar_max_1': coco_eval.stats[6],
        'ar_max_10': coco_eval.stats[7],
        'ar_max_100': coco_eval.stats[8],
        'ar_small': coco_eval.stats[9],
        'ar_medium': coco_eval.stats[10],
        'ar_large': coco_eval.stats[11]
    }
    
    return metrics

def calculate_metrics(gt, pred, score_thresholds):
    results = []
    for score_thresh in score_thresholds:
        pred_filtered = [p for p in pred if p['score'] >= score_thresh]
        if len(pred_filtered) == 0:
            continue

        count_accuracies = []
        errors = []
        gt_values = []
        pred_values = []

        gt_dict = {}
        for ann in gt:
            if ann['image_id'] not in gt_dict:
                gt_dict[ann['image_id']] = []
            gt_dict[ann['image_id']].append(ann)

        pred_dict = {}
        for ann in pred_filtered:
            if ann['image_id'] not in pred_dict:
                pred_dict[ann['image_id']] = []
            pred_dict[ann['image_id']].append(ann)

        for image_id in gt_dict.keys():
            gt_count = len(gt_dict[image_id])
            pred_count = len(pred_dict.get(image_id, []))
            count_accuracy = 1 - abs(pred_count / gt_count - 1)
            count_accuracies.append(count_accuracy)
            errors.append(gt_count - pred_count)
            gt_values.append(gt_count)
            pred_values.append(pred_count)

        rmse = np.sqrt(np.mean(np.square(errors)))
        avg_count_accuracy = np.mean(count_accuracies)
        sd_count_accuracy = np.std(count_accuracies)
        ss_total = np.sum(np.square(np.array(gt_values) - np.mean(gt_values)))
        ss_residual = np.sum(np.square(errors))
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else float('nan')
        mape = np.mean(np.abs(np.array(errors) / np.array(gt_values))) * 100

        results.append({
            'score_threshold': score_thresh,
            'r_squared': r_squared,
            'rmse': rmse,
            'mape': mape,
            'count_accuracy_avg': avg_count_accuracy,
            'count_accuracy_sd': sd_count_accuracy,
            'pred_counts': pred_values,
            'gt_counts': gt_values,
        })

    return results

def main():

    models_sizes = ['vits','vitb', 'vitl']
    shots = [1, 5, 10, 30, 50]
    dataset_numbers = [1, 2, 3]

    models = []

    for model_size in models_sizes:
        for shot in shots:
            for dataset_n in dataset_numbers:
                pred_path = f'output/{model_size}/{dataset_n}_{shot}shot/inference/coco_instances_results.json'
                gt_path = f'datasets/{dataset_n}/annotations/test.json'
                image_dir = f'datasets/{dataset_n}/test/'
                models.append({
                    'model': model_size,
                    'shots': shot,
                    'dataset': dataset_n,
                    'pred_path': pred_path,
                    'gt_path': gt_path,
                    'image_dir': image_dir
                })

    score_thresholds = [0, .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9, .98, .99]

    with open('metrics_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Shots', 'Dataset', 'Score Threshold', 'R-Squared', 'RMSE', 'MAPE', 
                        'Count Accuracy Avg', 'Count Accuracy SD', 'Pred Counts', 'GT Counts',
                        'AP (IoU=.50:.95)', 'AP (IoU=.50)', 
                        'AP (IoU=.75)', 'AP Small', 'AP Medium', 'AP Large', 'AR Max1', 'AR Max10', 
                        'AR Max100', 'AR Small', 'AR Medium', 'AR Large'])

        for model in models:
            with open(model['pred_path']) as f:
                predictions = json.load(f)

            with open(model['gt_path']) as f:
                gt_data = json.load(f)

            image_dict = {image['id']: image['file_name'] for image in gt_data['images']}
            gt_annotations = gt_data['annotations']

            pred_annotations = []
            for pred in predictions:
                pred_annotations.append({
                    'image_id': pred['image_id'],
                    'bbox': pred['bbox'],
                    'score': pred['score'],
                    'category_id': 1  # Ensure category_id is included for COCO evaluation
                })

            # Remap image_id with image names for count metrics calculation
            gt_annotations_remapped = []
            for ann in gt_annotations:
                ann['image_id'] = image_dict[ann['image_id']]
                gt_annotations_remapped.append(ann)

            pred_annotations_remapped = []
            for ann in pred_annotations:
                ann['image_id'] = image_dict[ann['image_id']]
                pred_annotations_remapped.append(ann)

            # Calculate custom counting metrics
            count_metrics = calculate_metrics(gt_annotations_remapped, pred_annotations_remapped, score_thresholds)
            
            print(f"Metrics for model {model['pred_path']}:")
            for score_thresh in score_thresholds:
                # Get count metrics for this threshold
                count_metric = next((m for m in count_metrics if m['score_threshold'] == score_thresh), None)
                if not count_metric:
                    continue
                
                # Calculate COCO metrics for this threshold
                try:
                    coco_metrics = calculate_coco_metrics(model['gt_path'], predictions, score_thresh)
                    
                    writer.writerow([
                        model['model'], model['shots'], model['dataset'], 
                        score_thresh, 
                        count_metric['r_squared'], 
                        count_metric['rmse'], 
                        count_metric['mape'],
                        count_metric['count_accuracy_avg'], 
                        count_metric['count_accuracy_sd'],
                        count_metric['pred_counts'],
                        count_metric['gt_counts'],
                        coco_metrics['ap_iou_50_95'],
                        coco_metrics['ap_iou_50'],
                        coco_metrics['ap_iou_75'],
                        coco_metrics['ap_small'],
                        coco_metrics['ap_medium'],
                        coco_metrics['ap_large'],
                        coco_metrics['ar_max_1'],
                        coco_metrics['ar_max_10'],
                        coco_metrics['ar_max_100'],
                        coco_metrics['ar_small'],
                        coco_metrics['ar_medium'],
                        coco_metrics['ar_large']
                    ])
                except Exception as e:
                    print(f"Error calculating COCO metrics at threshold {score_thresh}: {e}")
                    # Write row with just the count metrics and empty COCO metrics
                    writer.writerow([
                        model['model'], model['shots'], model['dataset'], 
                        score_thresh, 
                        count_metric['r_squared'], 
                        count_metric['rmse'], 
                        count_metric['mape'],
                        count_metric['count_accuracy_avg'], 
                        count_metric['count_accuracy_sd'],
                        count_metric['pred_counts'],
                        count_metric['gt_counts'],
                        '', '', '', '', '', '', '', '', '', '', '', ''
                    ])
                    
            # # Plot images with annotations
            # if model_size == 'vitl' and model['shots'] == 50 :

            #     plotted_images = set()
            #     for image_id, file_name in image_dict.items():
            #         if file_name not in plotted_images:
            #             image_path = os.path.join(model['image_dir'], file_name)
            #             gt_bboxes = [ann['bbox'] for ann in gt_annotations_remapped if ann['image_id'] == file_name]
            #             pred_bboxes = [ann['bbox'] for ann in pred_annotations_remapped if ann['image_id'] == file_name]
            #             pred_scores = [ann['score'] for ann in pred_annotations_remapped if ann['image_id'] == file_name]
            #             plot_bboxes(image_path, gt_bboxes, pred_bboxes, pred_scores)
            #             plotted_images.add(file_name)

if __name__ == "__main__":
    main()