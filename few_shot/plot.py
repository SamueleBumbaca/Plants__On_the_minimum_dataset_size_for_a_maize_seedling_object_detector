import json
import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from os.path import join
from os import makedirs

def plot_bboxes_1(image_path, gt_bboxes, pred_bboxes, pred_scores, plot_path):
    image = Image.open(image_path)
    
    # Create figure with two subplots side by side with additional space for colorbar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # First subplot - original image (no title)
    ax1.imshow(image)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Second subplot - annotated image (no title)
    ax2.imshow(image)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Plot ground truth annotations in red
    for bbox in gt_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=1, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

    # Plot predicted annotations in viridis color scale based on score
    norm = Normalize(vmin=0, vmax=1)
    for bbox, score in zip(pred_bboxes, pred_scores):
        color = cm.viridis(norm(score))
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=1, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)

    # Add colorbar to the right of the subplots
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    
    # Use specific positioning to minimize white space
    plt.subplots_adjust(left=0.01, right=0.87, bottom=0.01, top=0.99, wspace=0.02)
    
    cbar_ax = fig.add_axes([0.89, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Score')
    
    # Save the figure with tight bounding box to minimize white space
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_bboxes_2(image_path, gt_bboxes, pred_bboxes, pred_scores, plot_path):
    image = Image.open(image_path)
    
    # Create figure with two subplots side by side with additional space for colorbar
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # First subplot - original image (no title)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot ground truth annotations in red
    for bbox in gt_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # Plot predicted annotations in viridis color scale based on score
    norm = Normalize(vmin=0, vmax=1)
    for bbox, score in zip(pred_bboxes, pred_scores):
        color = cm.viridis(norm(score))
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
 
    # Save the figure with tight bounding box to minimize white space
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_bboxes_3(image_path, gt_bboxes, pred_bboxes, pred_scores, plot_path):
    image = Image.open(image_path)
    
    # Create figure with two subplots side by side with additional space for colorbar
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # First subplot - original image (no title)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot ground truth annotations in red
    for bbox in gt_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # Plot predicted annotations in viridis color scale based on score
    norm = Normalize(vmin=0, vmax=1)
    for bbox, score in zip(pred_bboxes, pred_scores):
        color = cm.viridis(norm(score))
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add score text near each predicted box
        ax.text(bbox[0], bbox[1]-5, f'{score:.2f}', 
                bbox=dict(facecolor=color, alpha=0.5, pad=0), 
                fontsize=8, color='white')
 
    # Save the figure with tight bounding box to minimize white space
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main(folder_path):

    # models_sizes = ['vits','vitb', 'vitl']
    # shots = [1, 5, 10, 30, 50]
    # dataset_numbers = [1, 2, 3]
    models_sizes = ['vitl']
    shots = [50]
    dataset_numbers = [1, 2, 3]
    score_threshold = {1: 0.1, 2: 0.3, 3: 0.5}

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
                
    for model in models:
        # print(f'Plotting images for {model["model"]} model with {model["shots"]} shot on dataset {model["dataset"]}')
        with open(model['pred_path']) as f:
            predictions = json.load(f)
        # print(f'Number of predictions: {len(predictions)}')
        with open(model['gt_path']) as f:
            gt_data = json.load(f)
        # print(f'Number of ground truth annotations: {len(gt_data["annotations"])}')
        image_dict = {image['id']: image['file_name'] for image in gt_data['images']}
        gt_annotations = gt_data['annotations']
        # print(f'Number of images: {len(image_dict)}')
        pred_annotations = []
        for pred in predictions:
            pred_annotations.append({
                'image_id': pred['image_id'],
                'bbox': pred['bbox'],
                'score': pred['score'],
                'category_id': 1  # Ensure category_id is included for COCO evaluation
            })
        # print(f'Number of images with predictions: {len(set([ann["image_id"] for ann in pred_annotations]))}')
        # Remap image_id with image names for count metrics calculation
        gt_annotations_remapped = []
        for ann in gt_annotations:
            ann['image_id'] = image_dict[ann['image_id']]
            gt_annotations_remapped.append(ann)
        # print(f'Number of images with ground truth annotations: {len(set([ann["image_id"] for ann in gt_annotations_remapped]))}')
        pred_annotations_remapped = []
        for ann in pred_annotations:
            ann['image_id'] = image_dict[ann['image_id']]
            pred_annotations_remapped.append(ann)
        # print(f'Number of images with predictions: {len(set([ann["image_id"] for ann in pred_annotations_remapped]))}')
        # Plot images with annotations
        plotted_images = set()
        for image_id, file_name in image_dict.items():
            if file_name not in plotted_images:
                plot_path = join(folder_path,model["model"],f'{model["dataset"]}_{model["shots"]}shot',f'{file_name}.pdf')
                if not os.path.exists(os.path.dirname(plot_path)):
                    os.makedirs(os.path.dirname(plot_path))

                image_path = os.path.join(model['image_dir'], file_name)
                gt_bboxes = [ann['bbox'] for ann in gt_annotations_remapped if ann['image_id'] == file_name]
                # If all models sizes, all shots and all datasets are plotted, then plot all predictions
                # pred_bboxes = [ann['bbox'] for ann in pred_annotations_remapped if ann['image_id'] == file_name]
                # pred_scores = [ann['score'] for ann in pred_annotations_remapped if ann['image_id'] == file_name]
                # If only one model size, one shot and one dataset is plotted, then plot predictions with score > threshold
                pred_bboxes = [ann['bbox'] for ann in pred_annotations_remapped if ann['image_id'] == file_name and ann['score'] > score_threshold[model['dataset']]]
                pred_scores = [ann['score'] for ann in pred_annotations_remapped if ann['image_id'] == file_name and ann['score'] > score_threshold[model['dataset']]]
                plot_bboxes_3(image_path, gt_bboxes, pred_bboxes, pred_scores,plot_path)
                plotted_images.add(file_name)

if __name__ == '__main__':
    folder_path = '/mnt/e/PhD/Paper2/plots'
    main(folder_path)