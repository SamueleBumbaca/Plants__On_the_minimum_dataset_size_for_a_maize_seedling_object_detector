import os
import json
import numpy as np
import csv
import torch
# import cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from os.path import basename

# HuggingFace imports
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

# Set the TORCH_CUDA_ARCH_LIST environment variable
os.environ['TORCH_CUDA_ARCH_LIST'] = "compute_86,sm_86"

# Function to perform inference using HuggingFace Grounding DINO
def evaluate_predictions(model, processor, image_path, prompt):
    # Open image
    image = Image.open(image_path).convert("RGB")
    
    # Process inputs without adding prefix
    text_prompt = prompt
    
    # Process inputs
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process outputs (get boxes, scores, labels)
    target_sizes = torch.tensor([image.size[::-1]])
    # Fix: Use the correct post-processing method for GroundingDino
    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.25  # Adjust threshold as needed
    )[0]  # Take first image result
    
    # Format results to match our evaluation pipeline format
    formatted_results = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        formatted_results.append({
            'boxes': box.unsqueeze(0),  # Add batch dimension
            'scores': score.unsqueeze(0),
            'score': score.item(),  # Explicit score key
            'labels': label.unsqueeze(0),
            'phrase': prompt  # Using the prompt as phrase
        })
    
    return formatted_results

# Function to convert annotations to COCO format
def geodataframe_to_coco(annotations, is_prediction=False, image_id=1):
    coco_annotations = []
    
    if is_prediction:
        # For predictions from Grounding DINO model
        for i, ann in enumerate(annotations):
            # Create bbox from the boxes tensor
            if 'boxes' in ann and ann['boxes'].numel() > 0:
                # Extract the box and ensure it's properly formatted
                box = ann['boxes'].cpu().numpy().tolist()
                
                # Debug print to see the actual box structure
                print(f"Box structure: {box}")
                
                # Check if box is a list of boxes or a single box
                if isinstance(box, list) and len(box) > 0:
                    # If it's a nested list, take the first box
                    if isinstance(box[0], list):
                        box = box[0]
                
                # Ensure the box has 4 coordinates
                if len(box) >= 4:
                    # Calculate area: width * height
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    area = width * height
                    
                    # Format box for COCO: [x, y, width, height]
                    coco_box = [box[0], box[1], width, height]
                    
                    coco_ann = {
                        "id": i,  # Generate an ID
                        "image_id": image_id,  # Use the provided image_id
                        "category_id": 1,  # Default to category 1
                        "bbox": coco_box,
                        "area": area,
                        "iscrowd": 0,
                        "score": ann.get("score", 0.0)
                    }
                    coco_annotations.append(coco_ann)
                else:
                    print(f"Warning: Box doesn't have 4 coordinates: {box}")
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

def compute_metrics(gt, pred, score_thresholds, prompt, image_id, image_file, dataset):
    results = []
    
    # Iterate over the score thresholds
    for score_thresh in score_thresholds:
        # Filter the predictions using the score threshold
        pred_filtered = [p for p in pred if p['score'] >= score_thresh]
        if len(pred_filtered) == 0:
            # If no predictions pass the threshold, create a result with default/zero values
            results.append({
                'model_id': 'groundingdino',
                'dataset': dataset,
                'image_id': image_id,
                'image_file': image_file,
                'prompt': prompt,
                'score_threshold': score_thresh,
                'r_squared': float('nan'),
                'rmse': float('nan'),
                'mape': float('nan'),
                'errors': [0],
                'gt_values': [0],
                'count_accuracy_avg': 0.0,
                'count_accuracy_sd': 0.0,
                'ap_iou_50_95': 0.0,
                'ap_iou_50': 0.0,
                'ap_iou_75': 0.0,
                'ap_small': 0.0,
                'ap_medium': 0.0,
                'ap_large': 0.0,
                'ar_max_1': 0.0,
                'ar_max_10': 0.0,
                'ar_max_100': 0.0,
                'ar_small': 0.0,
                'ar_medium': 0.0,
                'ar_large': 0.0
            })
            continue
            
        # Use the actual image_id from the annotations
        gt_group = [g for g in gt['annotations'] if g['image_id'] == image_id]
        pred_group = pred_filtered
        
        # Calculate count accuracy
        if len(gt_group) == 0:
            count_accuracy = 1.0 if len(pred_group) == 0 else 0.0
            errors = [0]
            gt_values = [0]
        else:
            count_accuracy = 1 - abs(len(pred_group) / len(gt_group) - 1)
            errors = [len(gt_group) - len(pred_group)]
            gt_values = [len(gt_group)]
            
        count_accuracies = [count_accuracy]
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
        if np.any(np.array(gt_values) == 0):
            mape = float('nan')  # Avoid division by zero
        else:
            mape = np.mean(np.abs(np.array(errors) / np.array(gt_values))) * 100
        
        # Create a reduced dataset with only the current image for evaluation
        current_image = None
        for img in gt['images']:
            if img['id'] == image_id:
                current_image = img
                break
        
        if not current_image:
            print(f"Warning: No image with ID {image_id} found in ground truth")
            continue
            
        # Convert ground truth and predictions to COCO format
        gt_annotations = [ann for ann in gt['annotations'] if ann['image_id'] == image_id]
        gt_annotations = geodataframe_to_coco(gt_annotations)
        pred_annotations = geodataframe_to_coco(pred_filtered, is_prediction=True, image_id=image_id)
        
        # Create COCO objects for this specific image only
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": image_id, "file_name": current_image.get('file_name', '')}],
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
            'model_id': 'groundingdino',
            'dataset': dataset,
            'image_id': image_id,
            'image_file': image_file,
            'prompt': prompt,
            'score_threshold': score_thresh,
            'r_squared': r_squared,
            'rmse': rmse,
            'mape': mape,
            'errors': errors,
            'gt_values': gt_values,
            'count_accuracy_avg': avg_count_accuracy,
            'count_accuracy_sd': sd_count_accuracy,
            'ap_iou_50_95': ap_metrics[0] if len(ap_metrics) > 0 else 0.0,
            'ap_iou_50': ap_metrics[1] if len(ap_metrics) > 1 else 0.0,
            'ap_iou_75': ap_metrics[2] if len(ap_metrics) > 2 else 0.0,
            'ap_small': ap_metrics[3] if len(ap_metrics) > 3 else 0.0,
            'ap_medium': ap_metrics[4] if len(ap_metrics) > 4 else 0.0,
            'ap_large': ap_metrics[5] if len(ap_metrics) > 5 else 0.0,
            'ar_max_1': ap_metrics[6] if len(ap_metrics) > 6 else 0.0,
            'ar_max_10': ap_metrics[7] if len(ap_metrics) > 7 else 0.0,
            'ar_max_100': ap_metrics[8] if len(ap_metrics) > 8 else 0.0,
            'ar_small': ap_metrics[9] if len(ap_metrics) > 9 else 0.0,
            'ar_medium': ap_metrics[10] if len(ap_metrics) > 10 else 0.0,
            'ar_large': ap_metrics[11] if len(ap_metrics) > 11 else 0.0
        })
    return results

# Function to save metrics to CSV
def save_metrics_to_csv(metrics, csv_filename):
    if not metrics:
        print(f"No metrics to save for {csv_filename}")
        # save as txt file
        with open(csv_filename, 'w') as f:
            f.write(str(metrics))
        print(f"Saved metrics to {csv_filename} as txt file")
        return
    keys = metrics[0].keys()
    with open(csv_filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(metrics)

def main(datasets_path, prompts, models, processors):
    # Iterate through the datasets
    for dataset in os.listdir(datasets_path):
        dataset_path = os.path.join(datasets_path, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        annotations_path = os.path.join(dataset_path, "annotations", "test.json")
        if not os.path.exists(annotations_path):
            print(f"Skipping {dataset}: annotations file not found at {annotations_path}")
            continue
            
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        all_metrics = []

        for prompt in prompts:
            for image_info in annotations["images"]:
                image_id = image_info["id"]
                image_file = image_info["file_name"]
                image_path = os.path.join(datasets_path, dataset, "test", image_file)
                
                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Skipping {image_file}: file not found at {image_path}")
                    continue
                
                # Perform inference and evaluate predictions for each model
                for model, processor in zip(models, processors):
                    results = evaluate_predictions(model, processor, image_path, prompt)
                    print(f"Results for {prompt} prompt on {image_file}: {results}")
                    metrics = compute_metrics(annotations, results, [0.25], prompt, image_id, image_file, dataset)
                    all_metrics.extend(metrics)
        
        # Save all metrics to CSV
        csv_filename = os.path.join(datasets_path, dataset, "metrics_groundingdino.csv")
        save_metrics_to_csv(all_metrics, csv_filename)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model using specific model class
    try:
        model_id = "IDEA-Research/grounding-dino-base"
        print(f"Loading model from {model_id}...")
        
        # Use specific model classes instead of Auto classes
        processor = GroundingDinoProcessor.from_pretrained(model_id)
        model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
        print("Grounding DINO model loaded successfully from HuggingFace")
    except Exception as e:
        print(f"Error loading Grounding DINO model: {e}")
        try:
            # Try alternative repository
            model_id = "facebook/detr-resnet-50"
            print(f"Trying alternative model (DETR): {model_id}...")
            
            from transformers import DetrImageProcessor, DetrForObjectDetection
            processor = DetrImageProcessor.from_pretrained(model_id)
            model = DetrForObjectDetection.from_pretrained(model_id).to(device)
            print("DETR model loaded successfully as fallback")
        except Exception as e2:
            print(f"Error loading alternative model: {e2}")
            model = None
            processor = None

    # Define the datasets path
    datasets_path = "datasets"

    # Define prompts for evaluation
    prompts = ["plant", "maize", "seedling"]
    
    if model is not None and processor is not None:
        main(datasets_path, prompts, [model], [processor])
    else:
        print("Model loading failed. Exiting.")