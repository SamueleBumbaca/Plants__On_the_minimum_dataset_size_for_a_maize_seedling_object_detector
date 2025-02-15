import rasterio
import json
from os.path import join
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_patches_and_annotations(session_folder, transformed_annotation_path):
    # Load transformed annotations
    with open(transformed_annotation_path) as f:
        coco_annotations = json.load(f)
    
    patches_folder = join(session_folder, 'patches')
    
    for patch_file in os.listdir(patches_folder):
        patch_path = join(patches_folder, patch_file)
        
        with rasterio.open(patch_path) as src:
            patch_image = src.read([1, 2, 3])
            fig, ax = plt.subplots(1)
            ax.imshow(patch_image.transpose(1, 2, 0))
            
            patch_transform = src.transform
            for annotation in coco_annotations['annotations']:
                bbox = annotation['bbox']
                x, y, width, height = bbox
                
                # Check if the bbox is within the patch
                patch_x, patch_y = ~patch_transform * (x, y)
                if 0 <= patch_x < patch_image.shape[1] and 0 <= patch_y < patch_image.shape[0]:
                    patch_width, patch_height = width / patch_transform.a, height / patch_transform.e
                    rect = patches.Rectangle((patch_x, patch_y), patch_width, patch_height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
            
            plt.show()

# Example usage
session_folder = 'dataset/DAVIDetAL/1_hermine_2019_1'
transformed_annotation_path = join(session_folder, 'transformed_annotations.json')
plot_patches_and_annotations(session_folder, transformed_annotation_path)