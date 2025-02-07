import fiona
from shapely.geometry import shape, mapping
import numpy as np
from tqdm import tqdm
import torch
from torchvision.ops import nms

def read_shapefile(shapefile_path):
    with fiona.open(shapefile_path, 'r') as shp:
        crs = shp.crs
        schema = shp.schema
        features = [feature for feature in shp]
    return crs, schema, features

def write_shapefile(shapefile_path, crs, schema, features):
    with fiona.open(shapefile_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as shp:
        for feature in features:
            shp.write(feature)

def filter_boxes_with_nms(input_shapefile, output_shapefile, max_dist=0.03, iou_threshold=0.5):
    crs, schema, features = read_shapefile(input_shapefile)
    
    boxes = []
    scores = []
    centroids = []
    for feature in tqdm(features, desc="Processing features"):
        geom = shape(feature['geometry'])
        minx, miny, maxx, maxy = geom.bounds
        centroid = geom.centroid
        boxes.append([minx, miny, maxx, maxy])
        scores.append(feature['properties']['score'])
        centroids.append([centroid.x, centroid.y])
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    centroids = np.array(centroids)
    
    # Merge arrays
    merged_array = np.hstack((boxes, scores[:, np.newaxis], centroids))
    
    # Sort by centroid dimensions in descending order
    sorted_array = merged_array[np.lexsort((-merged_array[:, -1], -merged_array[:, -2]))]
    
    filtered_features = []
    while len(sorted_array) > 0:
        interval_boxes = []
        interval_scores = []
        interval_indices = []
        
        # Calculate distances from the first centroid to all other centroids
        distances = np.linalg.norm(sorted_array[:, -2:] - sorted_array[0, -2:], axis=1)
        
        # Select boxes and scores where distance is less than or equal to max_dist
        within_dist_indices = np.where(distances <= max_dist)[0]
        
        interval_boxes = sorted_array[within_dist_indices, :4]
        interval_scores = sorted_array[within_dist_indices, 4]
        
        interval_boxes = torch.tensor(interval_boxes, dtype=torch.float32)
        interval_scores = torch.tensor(interval_scores, dtype=torch.float32)
        
        keep_indices = nms(interval_boxes, interval_scores, iou_threshold).cpu().numpy()
        
        for idx in keep_indices:
            feature_idx = within_dist_indices[idx]
            feature = features[feature_idx]
            filtered_features.append(feature)
        
        # Remove the selected observations from the sorted array
        sorted_array = np.delete(sorted_array, within_dist_indices, axis=0)
    
    # Update schema to include score attribute
    schema['properties']['score'] = 'float'
    
    write_shapefile(output_shapefile, crs, schema, filtered_features)

if __name__ == '__main__':
    import click

    @click.command()
    @click.option('-in', '--input_shapefile', required=True, type=str, help='Path to the input shapefile with overlapping boxes')
    @click.option('-out', '--output_shapefile', required=True, type=str, help='Path to the output shapefile with filtered boxes')
    @click.option('-dist', '--max_dist', default=(0.18+0.12)/2/0.005, type=float, help='Maximum distance for clustering centroids')
    @click.option('-iou', '--iou_threshold', default=0.5, type=float, help='IoU threshold for Non-Maximum Suppression')
    def main(input_shapefile, output_shapefile, max_dist, iou_threshold):
        print(f'Reading shapefile from {input_shapefile}')
        filter_boxes_with_nms(input_shapefile, output_shapefile, max_dist, iou_threshold)
        print(f'Filtered shapefile saved to {output_shapefile}')

    main()