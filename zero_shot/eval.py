import os
import json
import numpy as np
import csv
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from os.path import basename

# Set the TORCH_CUDA_ARCH_LIST environment variable
os.environ['TORCH_CUDA_ARCH_LIST'] = "compute_86,sm_86"

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib import cm

def save_plot(annotations, predictions, image_path, save_plot_path):
    """
    Create visualization of ground truth and predicted bounding boxes
    
    Args:
        annotations: COCO format annotations dict
        predictions: List of prediction dicts with 'boxes', 'scores', etc.
        image_path: Path to the image file
        save_plot_path: Where to save the output plot
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Get image_id from the filename to find the corresponding ground truth
    image_filename = os.path.basename(image_path)
    image_id = None
    for img_info in annotations["images"]:
        if img_info["file_name"] == image_filename:
            image_id = img_info["id"]
            break
    
    if image_id is None:
        print(f"Warning: Could not find image_id for {image_filename}")
        return
    
    # Extract ground truth bounding boxes for this image
    gt_bboxes = []
    for ann in annotations["annotations"]:
        if ann["image_id"] == image_id:
            # COCO format is [x, y, width, height]
            gt_bboxes.append(ann["bbox"])
    
    # Extract predicted bounding boxes and scores
    pred_bboxes = []
    pred_scores = []
    for pred in predictions:
        if "boxes" in pred and pred["boxes"].numel() > 0:
            # Convert tensor to numpy and convert to [x, y, width, height] format
            boxes = pred["boxes"].cpu().numpy()
            if len(boxes.shape) == 1:
                # Single box case
                x1, y1, x2, y2 = boxes
                pred_bboxes.append([x1, y1, x2-x1, y2-y1])
                pred_scores.append(pred.get("score", 0.5))
            else:
                # Multiple boxes case
                for box_idx in range(boxes.shape[0]):
                    x1, y1, x2, y2 = boxes[box_idx]
                    pred_bboxes.append([x1, y1, x2-x1, y2-y1])
                    if "scores" in pred and len(pred["scores"]) > box_idx:
                        pred_scores.append(pred["scores"][box_idx].item())
                    else:
                        pred_scores.append(pred.get("score", 0.5))
    
    # Create figure for plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image (without title)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot ground truth annotations in black
    for bbox in gt_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=2, edgecolor='black', facecolor='none',
                                label='Ground Truth')
        ax.add_patch(rect)

    # Plot predicted annotations in viridis color scale based on score
    norm = Normalize(vmin=0, vmax=1)
    for bbox, score in zip(pred_bboxes, pred_scores):
        color = cm.viridis(norm(score))
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # Add score text near the box
        #ax.text(bbox[0], bbox[1]-5, f'{score:.2f}', 
        #        bbox=dict(facecolor=color, alpha=0.5, pad=0), 
        #        fontsize=8, color='white')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# Function to perform inference and evaluate predictions
def predict(model, processor, image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    target_sizes = torch.tensor([(image.height, image.width)])
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1
    )
    # Ensure result contains 'score' key
    for res in result:
        if 'score' not in res:
            res['score'] = res['scores'][0].item() if res['scores'].numel() > 0 else 0.0
    return result

# Function to convert annotations to COCO format
def geodataframe_to_coco(annotations, is_prediction=False, image_id=1):
    coco_annotations = []
    
    if is_prediction:
        # For predictions from OWLv2 model
        for i, ann in enumerate(annotations):
            if 'bbox' in ann:
                box = ann['bbox']
                
                # For tensors, convert to list
                if torch.is_tensor(box):
                    box = box.cpu().numpy().tolist()
                
                # Ensure the box has 4 coordinates
                if len(box) >= 4:
                    # OWLv2 always returns [x1, y1, x2, y2] format
                    # Convert to COCO format [x, y, width, height]
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    area = width * height
                    
                    # COCO format box
                    coco_box = [x1, y1, width, height]
                    
                    # Convert score to Python primitive if it's a tensor
                    score = ann.get("score", 0.0)
                    if torch.is_tensor(score):
                        score = score.item()
                    
                    coco_ann = {
                        "id": i,
                        "image_id": image_id,
                        "category_id": ann.get("category_id", 1),
                        "bbox": coco_box,
                        "area": area,
                        "iscrowd": 0,
                        "score": score
                    }
                    coco_annotations.append(coco_ann)
    else:
        # For ground truth annotations
        for ann in annotations:
            if all(k in ann for k in ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]):
                coco_ann = {
                    "id": ann["id"],
                    "image_id": ann["image_id"],
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"]
                }
                coco_annotations.append(coco_ann)
    
    return coco_annotations

def compute_metrics(gt, predictions, score_thresholds, prompt, dataset, card):
    results = []
    for image_info in gt["images"]:
        image_id = image_info["id"]
        image_file = image_info["file_name"]
        pred = predictions[image_file][0]
        # Iterate over the score thresholds
        for score_thresh in score_thresholds:
            # Filter the predictions using the score threshold
            pred_index = pred['scores']>score_thresh
            # Convert tensor to Python int
            pred_len = sum(pred_index).item() if torch.is_tensor(sum(pred_index)) else sum(pred_index)
            gt_group = [g for g in gt['annotations'] if g['image_id'] == image_id]
            gt_len = len(gt_group)
            if pred_len == 0:
                # If no predictions pass the threshold, create a result with default/zero values
                results.append({
                    'model': card,
                    'threshold': score_thresh,
                    'dataset': dataset,
                    'image_id': image_id,
                    'image_file': image_file,
                    'prompt': prompt,
                    'pred_len': pred_len,
                    'gt_len': gt_len,
                    'count_accuracy': np.nan,
                    'ap_iou_50_95': np.nan,
                    'ap_iou_50': np.nan,
                    'ap_iou_75': np.nan,
                    'ap_small': np.nan,
                    'ap_medium': np.nan,
                    'ap_large': np.nan,
                    'ar_max_1': np.nan,
                    'ar_max_10': np.nan,
                    'ar_max_100': np.nan,
                    'ar_small': np.nan,
                    'ar_medium': np.nan,
                    'ar_large': np.nan,
                })
                continue

            pred_scores = pred['scores'][pred_index].cpu()
            pred_boxes = pred['boxes'][pred_index].cpu()
            pred_labels = pred['labels'][pred_index].cpu()
            pred_group = [{'image_id': image_id, 'category_id': 1, 'bbox': box, 'score': score} for box, score in zip(pred_boxes, pred_scores)]
            # Calculate count accuracy
            if gt_len == 0:
                count_accuracy = 1.0 if pred_len == 0 else 0.0
                pred_len = 0
            else:
                # Convert tensor to Python float
                count_accuracy_tensor = 1 - abs(pred_len / gt_len - 1)
                count_accuracy = count_accuracy_tensor.item() if torch.is_tensor(count_accuracy_tensor) else count_accuracy_tensor
           
            # Convert ground truth and predictions to COCO format
            gt_annotations = [ann for ann in gt['annotations'] if ann['image_id'] == image_id]
            gt_annotations = geodataframe_to_coco(gt_annotations)
            pred_annotations = geodataframe_to_coco(pred_group, is_prediction=True, image_id=image_id)
            
            # Create COCO objects for this specific image only
            coco_gt = COCO()
            coco_gt.dataset = {
                "images": [{"id": image_id, "file_name": image_file}],
                "annotations": gt_annotations,
                "categories": [{"id": 1, "name": "object"}]
            }
            coco_gt.createIndex()
            
            # Check if there are any predictions
            if len(pred_annotations) == 0:
                # No predictions, set all AP/AR metrics to 0
                ap_metrics = [0.0] * 12
            else:
                # Load predictions and evaluate
                try:
                    coco_pred = coco_gt.loadRes(pred_annotations)
                    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    ap_metrics = coco_eval.stats
                except Exception as e:
                    print(f"Error during COCO evaluation: {e}")
                    ap_metrics = [0.0] * 12
                
            # Save the results
            results.append({
                'model': card,
                'threshold': score_thresh,
                'dataset': dataset,
                'image_id': image_id,
                'image_file': image_file,
                'prompt': prompt,
                'pred_len': pred_len,
                'gt_len': gt_len,
                'count_accuracy': count_accuracy,
                'ap_iou_50_95': ap_metrics[0] if len(ap_metrics) > 0 else np.nan,
                'ap_iou_50': ap_metrics[1] if len(ap_metrics) > 1 else np.nan,
                'ap_iou_75': ap_metrics[2] if len(ap_metrics) > 2 else np.nan,
                'ap_small': ap_metrics[3] if len(ap_metrics) > 3 else np.nan,
                'ap_medium': ap_metrics[4] if len(ap_metrics) > 4 else np.nan,
                'ap_large': ap_metrics[5] if len(ap_metrics) > 5 else np.nan,
                'ar_max_1': ap_metrics[6] if len(ap_metrics) > 6 else np.nan,
                'ar_max_10': ap_metrics[7] if len(ap_metrics) > 7 else np.nan,
                'ar_max_100': ap_metrics[8] if len(ap_metrics) > 8 else np.nan,
                'ar_small': ap_metrics[9] if len(ap_metrics) > 9 else np.nan,
                'ar_medium': ap_metrics[10] if len(ap_metrics) > 10 else np.nan,
                'ar_large': ap_metrics[11] if len(ap_metrics) > 11 else np.nan,
            })
    return results

# Function to save metrics to CSV
def save_metrics_to_csv(metrics, csv_filename):
    if not metrics:
        print(f"No metrics to save for {csv_filename}")
        # save as txt file
        with open(csv_filename, 'w') as f:
            f.write(metrics)
        print(f"Saved metrics to {csv_filename} as txt file")
        return
    keys = metrics[0].keys()
    with open(csv_filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(metrics)

def main(datasets_path, prompts, models, processors, cards):
    all_metrics = []
    score_thresholds = [0, .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9, .98, .99]
    # Iterate through the datasets
    for dataset in os.listdir(datasets_path):
        annotations_path = os.path.join(datasets_path, dataset, "annotations", "test.json")
        with open(annotations_path) as f:
            annotations = json.load(f)

        for prompt in prompts:
            # Perform inference and evaluate predictions for each model
            for model, processor, card in zip(models, processors, cards):

                results = {}
                for image_info in annotations["images"]:
                    image_id = image_info["id"]
                    image_file = image_info["file_name"]
                    image_path = os.path.join(datasets_path, dataset, "test", image_file)
                    result = predict(model, processor, image_path, prompt)
                    print(f"Results for {prompt} prompt on {image_file}: {result}")
                    save_plot_path = os.path.join("E:/PhD/Paper2/plots/OWLv2", f"{prompt}_{image_file}_{card}.png")
                    save_plot(annotations, result, image_path, save_plot_path)
                    results[image_info["file_name"]] = result
                    
                metrics = compute_metrics(annotations, results, score_thresholds, prompt, dataset, card)
                all_metrics.extend(metrics)
        
    # Save all metrics to CSV
    csv_filename = os.path.join("metrics.csv")
    save_metrics_to_csv(all_metrics, csv_filename)

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    cards = [
        "google/owlv2-base-patch16",
        "google/owlv2-base-patch16-ensemble",
        "google/owlv2-base-patch16-finetuned",
        "google/owlv2-large-patch14",
        "google/owlv2-large-patch14-ensemble",
        "google/owlv2-large-patch14-finetuned",
    ]
    models = []
    processors = []

    for card in cards:
        # Load models and processors
        try:
            processor = Owlv2Processor.from_pretrained(card)
            model = Owlv2ForObjectDetection.from_pretrained(card).to(device)
            processors.append(processor)
            models.append(model)
        except Exception as e:
            print(f"Error loading Owlv2 model: {e}")

    cards = [basename(card) for card in cards]

    # Define the datasets path
    datasets_path = "datasets"

    # Define prompts for evaluation
    prompts = [
        "plant", "maize", "seedling",
        "aerial view of maize seedlings",
        "corn seedlings in rows",
        "young maize plants from above", 
        "crop rows with corn seedlings",
        "maize seedlings with regular spacing",
        "top-down view of corn plants",
        "agricultural field with maize seedlings",
        "orthomosaic of corn plants in rows"
    ]
        
    main(datasets_path, prompts, models, processors, cards)