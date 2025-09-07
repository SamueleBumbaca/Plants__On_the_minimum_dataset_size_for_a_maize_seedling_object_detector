import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the JSON data
with open('/home/samuelebumbaca/repositories/CDFSOD-benchmark/datasets/3/annotations/train.json') as f:
    data = json.load(f)

# Create a dictionary to map image ids to file names
image_dict = {image['id']: image['file_name'] for image in data['images']}

# Create a dictionary to store annotations by image id
annotations_dict = {}
for annotation in data['annotations']:
    image_id = annotation['image_id']
    if image_id not in annotations_dict:
        annotations_dict[image_id] = []
    annotations_dict[image_id].append(annotation)

# Directory where images are stored
image_dir = '/home/samuelebumbaca/repositories/CDFSOD-benchmark/datasets/3/train/'

# Plot the bounding boxes on the images
for image_id, file_name in image_dict.items():
    image_path = os.path.join(image_dir, file_name)
    image = Image.open(image_path)
    print(file_name)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if image_id in annotations_dict:
        for annotation in annotations_dict[image_id]:
            bbox = annotation['bbox']
            # Assuming bbox format is [x, y, width, height]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            print(bbox)

    plt.show()