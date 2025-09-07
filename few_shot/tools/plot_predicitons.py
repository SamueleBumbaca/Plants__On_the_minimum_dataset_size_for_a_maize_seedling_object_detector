import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

# Load the JSON data for predictions and test annotations
with open('/home/samuelebumbaca/repositories/CDFSOD-benchmark/output/vitl/1_10shot/inference/coco_instances_results.json') as f:
    predictions = json.load(f)

with open('/home/samuelebumbaca/repositories/CDFSOD-benchmark/datasets/1/annotations/test.json') as f:
    test_data = json.load(f)

# Create a dictionary to map image ids to file names
image_dict = {image['id']: image['file_name'] for image in test_data['images']}

# Create a dictionary to store ground truth annotations by image id
gt_annotations_dict = {}
for annotation in test_data['annotations']:
    image_id = annotation['image_id']
    if image_id not in gt_annotations_dict:
        gt_annotations_dict[image_id] = []
    gt_annotations_dict[image_id].append(annotation)

# Create a dictionary to store predicted annotations by image id
pred_annotations_dict = {}
for prediction in predictions:
    image_id = prediction['image_id']
    if image_id not in pred_annotations_dict:
        pred_annotations_dict[image_id] = []
    pred_annotations_dict[image_id].append(prediction)

# Directory where images are stored
image_dir = '/home/samuelebumbaca/repositories/CDFSOD-benchmark/datasets/1/test/'

# Plot the bounding boxes on the images
for image_id, file_name in image_dict.items():
    image_path = os.path.join(image_dir, file_name)
    image = Image.open(image_path)
    print(file_name)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot ground truth annotations in red
    if image_id in gt_annotations_dict:
        for annotation in gt_annotations_dict[image_id]:
            bbox = annotation['bbox']
            # Assuming bbox format is [x, y, width, height]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            print(f"GT bbox: {bbox}")

    # Plot predicted annotations in viridis color scale based on score
    if image_id in pred_annotations_dict:
        for prediction in pred_annotations_dict[image_id]:
            bbox = prediction['bbox']
            score = prediction['score']
            color = cm.viridis(score)  # Get color from viridis colormap based on score
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            print(f"Pred bbox: {bbox}, score: {score}")

    plt.show()