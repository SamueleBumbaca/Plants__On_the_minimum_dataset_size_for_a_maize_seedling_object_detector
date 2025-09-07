import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from ultralytics import RTDETR
import cv2
from pathlib import Path
import glob
import os

def read_yolo_labels(txt_path, img_shape):
    """
    Reads YOLO-format labels and returns a list of [x1, y1, x2, y2] in pixel coordinates.
    """
    h, w = img_shape[:2]
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # YOLO: class cx cy w h (all normalized)
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
    return boxes

def find_model_file(model_path):
    """Find the actual .pt model file in the directory"""
    model_dir = Path(model_path)
    
    # Look for .pt files in the directory
    pt_files = list(model_dir.glob("*.pt"))
    if pt_files:
        # Return the first .pt file found (usually best.pt or last.pt)
        best_pt = [f for f in pt_files if 'best' in f.name.lower()]
        if best_pt:
            return best_pt[0]
        return pt_files[0]
    
    # Also check subdirectories (weights folder)
    weights_dir = model_dir / "weights"
    if weights_dir.exists():
        pt_files = list(weights_dir.glob("*.pt"))
        if pt_files:
            best_pt = [f for f in pt_files if 'best' in f.name.lower()]
            if best_pt:
                return best_pt[0]
            return pt_files[0]
    
    return None

def load_image(image_path):
    """Load and preprocess image for prediction"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def predict_with_raw_outputs(model, image_path):
    """Predict with model and return raw outputs before NMS and thresholding"""
    # Load image
    image = load_image(image_path)
    
    # Perform prediction with raw outputs
    results = model.predict(source=image_path, save=False, verbose=False)
    
    # Get raw predictions (before NMS and thresholding)
    raw_results = results[0]
    
    # Extract boxes and confidences
    if hasattr(raw_results, 'boxes') and raw_results.boxes is not None:
        boxes = raw_results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = raw_results.boxes.conf.cpu().numpy()
        return image, boxes, confidences
    else:
        return image, np.array([]), np.array([])

def plot_predictions(image, boxes, confidences, ax, gt_boxes=None):
    """Plot image with GT and predicted bounding boxes."""
    ax.imshow(image)
    ax.axis('off')

    # Plot ground truth boxes first (in black, below predictions)
    if gt_boxes is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='black', facecolor='none', zorder=1
            )
            ax.add_patch(rect)

    # Plot predicted boxes (in viridis, above GT)
    if len(boxes) > 0 and len(confidences) > 0:
        norm_conf = np.clip(confidences, 0, 1)
        cmap = plt.cm.viridis
        for box, conf in zip(boxes, norm_conf):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            color = cmap(conf)
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none', zorder=2
            )
            ax.add_patch(rect)

def main():
    # Define model and image pairs (corrected paths)
    predictions_config = [
        {
            'model_path': 'experiments/models/dataset_size/rtdetr-l_dataset_2_exp_dataset_size_67',
            'image_path': 'dataset/2/yolo_datasets/DatasetSize_10_Train_9_Val_1/val/SAGIT22_C_4178_2022_04_29-34-14.tif',
            'title': 'Dataset 2 - RT-DETR-L (Exp 67)'
        },
        {
            'model_path': 'experiments/models/dataset_size/rtdetr-l_dataset_3_exp_dataset_size_68',
            'image_path': 'dataset/3/yolo_datasets/DatasetSize_10_Train_9_Val_1/val/SAGIT22_C_4178_2022_05_10-51-27.tif',  # Fixed: dataset/3 instead of dataset/1
            'title': 'Dataset 3 - RT-DETR-L (Exp 68)'
        },
        {
            'model_path': 'experiments/models/dataset_size/rtdetr-l_dataset_1_exp_dataset_size_69',
            'image_path': 'dataset/1/yolo_datasets/DatasetSize_10_Train_9_Val_1/val/SAGIT22_C_4175_2022_05_27-39-14.tif',
            'title': 'Dataset 1 - RT-DETR-L (Exp 69)'
        }
    ]
    
    # Create output directory
    output_dir = Path('prediction_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create PDF for saving results
    pdf_path = output_dir / 'rtdetr_predictions_raw.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        for idx, config in enumerate(predictions_config):
            print(f"Processing {config['title']}...")
            
            try:
                # Find the actual model file
                model_file = find_model_file(config['model_path'])
                if model_file is None:
                    print(f"No .pt model file found in: {config['model_path']}")
                    # List contents of directory for debugging
                    model_dir = Path(config['model_path'])
                    if model_dir.exists():
                        print(f"  Directory contents: {list(model_dir.iterdir())}")
                    continue
                
                print(f"  Using model: {model_file}")
                model = RTDETR(str(model_file))
                
                # Check if image exists
                image_path = Path(config['image_path'])
                if not image_path.exists():
                    print(f"Image not found: {config['image_path']}")
                    # Try to find alternative image formats
                    alt_formats = ['.jpg', '.png', '.jpeg']
                    found = False
                    for ext in alt_formats:
                        alt_path = image_path.with_suffix(ext)
                        if alt_path.exists():
                            image_path = alt_path
                            print(f"  Found alternative: {image_path}")
                            found = True
                            break
                    
                    if not found:
                        continue
                
                # Get predictions
                image, boxes, confidences = predict_with_raw_outputs(model, str(image_path))
                
                # Find ground truth label file
                gt_path = image_path.with_suffix('.txt')
                gt_boxes = read_yolo_labels(str(gt_path), image.shape)
                
                # Plot results
                ax = axes[idx]
                plot_predictions(image, boxes, confidences, ax, gt_boxes=gt_boxes)

                print(f"  - Found {len(boxes)} detections")
                if len(confidences) > 0:
                    print(f"  - Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
                
            except Exception as e:
                print(f"Error processing {config['title']}: {str(e)}")
                # Plot empty image with error message
                ax = axes[idx]
                ax.text(0.5, 0.5, f"Error: {str(e)}", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='red')
                ax.set_title(config['title'] + " (ERROR)", fontsize=12)
                ax.axis('off')
        
        # Adjust layout and save
        plt.subplots_adjust(hspace=0.3)
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"\nResults saved to: {pdf_path}")
    
    # Also save individual images
    for idx, config in enumerate(predictions_config):
        try:
            model_file = find_model_file(config['model_path'])
            if model_file is None:
                continue
            
            model = RTDETR(str(model_file))
            
            image_path = Path(config['image_path'])
            if not image_path.exists():
                # Try alternative formats
                alt_formats = ['.jpg', '.png', '.jpeg']
                found = False
                for ext in alt_formats:
                    alt_path = image_path.with_suffix(ext)
                    if alt_path.exists():
                        image_path = alt_path
                        found = True
                        break
                if not found:
                    continue
            
            image, boxes, confidences = predict_with_raw_outputs(model, str(image_path))
            
            # Save individual plot
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            plot_predictions(image, boxes, confidences, config['title'], ax)
            
            output_file = output_dir / f"prediction_{idx+1}.pdf"
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Individual plot saved: {output_file}")
            
        except Exception as e:
            print(f"Error saving individual plot for {config['title']}: {str(e)}")

if __name__ == "__main__":
    main()