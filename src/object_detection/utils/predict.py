from object_detection.models import DETR, YOLO
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import nms
import numpy as np
import fiona
import rasterio
from rasterio.mask import mask
from shapely.geometry import box as shapely_box, mapping, shape, Point
from shapely.affinity import affine_transform
from PIL import Image
from skimage import io
import yaml
import glob
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(cfg,ckpt):
    if 'yolo' in cfg['train']['model']:
        print('Loading YOLO')
        model = YOLO(ckpt) 
    elif 'detr' in cfg['train']['model']:
        print('Load DETR')
        model = DETR(ckpt)
    else:
        raise f'Not implemented model {cfg['train']['model']}'
    return model

def get_test_tiles(shp_path, raster_path, field_shp, output_folder = 'temp'):
    def write_tile(output_folder, name, polygon, src):
        # Crop the raster with the polygon
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()
        # Update the metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        # Save the cropped raster
        output_path = os.path.join(output_folder, name)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        return output_path, polygon
    
    def intersect_polygons(tiles, field):
        with fiona.open(field, 'r') as shapefile:
            field_shape = shape(shapefile[0]['geometry'])
        with fiona.open(tiles, 'r') as shapefile:
            tiles = [feature for feature in shapefile if shape(feature['geometry']).intersects(field_shape)]
        return tiles

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    # Open the shapefile
    with fiona.open(shp_path, 'r') as shapefile:
        # Get the CRS of the shapefile
        shp_crs = shapefile.crs
        # Open the raster file
        with rasterio.open(raster_path) as src:
            # Get the CRS of the raster
            raster_crs = src.crs
            # Check if the CRS match
            if shp_crs != raster_crs:
                raise ValueError("CRS of the shapefile and raster do not match")
            shapefile = intersect_polygons(shp_path, field_shp)
            # List to store the paths of the new rasters
            new_raster_paths = []
            # Iterate over each polygon in the shapefile
            geometries = {}
            for feature in shapefile:
                # Write the tile to the output folder
                polygon = shape(feature['geometry'])
                # Calculate the width of the bounding box
                minx, miny, maxx, maxy = polygon.bounds
                width = maxx - minx
                name = feature['properties']['NAME']
                output_path, polygon = write_tile(output_folder, name, polygon, src)
                # Add the path to the list
                new_raster_paths.append(output_path)
                geometries[name] = polygon
                # Write the tile with right shift coordinates
                # Translate the bounding box to the right by half of its width
                right_polygon = shapely_box(minx + width / 2, miny, maxx + width / 2, maxy)
                right_name = name + '_right'
                right_output_path, right_polygon = write_tile(output_folder, right_name, right_polygon, src)
                new_raster_paths.append(right_output_path)
                geometries[right_name] = right_polygon
                # Write the tile with left shift coordinates
                # Translate the bounding box to the left by half of its width
                left_polygon = shapely_box(minx - width / 2, miny, maxx - width / 2, maxy)
                left_name = name + '_left'
                left_output_path, left_polygon = write_tile(output_folder, left_name, left_polygon, src)
                new_raster_paths.append(left_output_path)
                geometries[left_name] = left_polygon
                #Write the tile with up shift coordinates
                # Translate the bounding box up by half of its height
                up_polygon = shapely_box(minx, miny + width / 2, maxx, maxy + width / 2)
                up_name = name + '_up'
                up_output_path, up_polygon = write_tile(output_folder, up_name, up_polygon, src)
                new_raster_paths.append(up_output_path)
                geometries[up_name] = up_polygon
                #Write the tile with up left shift coordinates
                # Translate the bounding box up left by half of its height and width
                up_left_polygon = shapely_box(minx - width / 2, miny + width / 2, maxx - width / 2, maxy + width / 2)
                up_left_name = name + '_up_left'
                up_left_output_path, up_left_polygon = write_tile(output_folder, up_left_name, up_left_polygon, src)
                new_raster_paths.append(up_left_output_path)
                geometries[up_left_name] = up_left_polygon
                #Write the tile with up right shift coordinates
                # Translate the bounding box up right by half of its height and width
                up_right_polygon = shapely_box(minx + width / 2, miny + width / 2, maxx + width / 2, maxy + width / 2)
                up_right_name = name + '_up_right'
                up_right_output_path, up_right_polygon = write_tile(output_folder, up_right_name, up_right_polygon, src)
                new_raster_paths.append(up_right_output_path)
                geometries[up_right_name] = up_right_polygon
    return new_raster_paths, geometries

class CustomTileDataset(Dataset):
    def __init__(self, tile_paths, transform=None):
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        img_path = self.tile_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def predict_bounding_boxes(model, dataloader):
    all_predictions = {}
    for images, path in dataloader:
        tile_name = os.path.basename(path[0])
        pred = model(images)
        boxes = pred[0].boxes.xyxy.cpu().numpy()  # Convert tensor to numpy array on CPU
        boxes = torch.tensor(boxes).to('cuda')
        scores = pred[0].boxes.conf.cpu().numpy()  # Confidence scores
        scores = torch.tensor(scores).to('cuda')
        labels = pred[0].boxes.cls.cpu().numpy()  # Class labels
        labels = torch.tensor(labels).to('cuda')
        all_predictions[tile_name] = {'tile_path' : path,
                             'image_coords' : [{
            'boxes': boxes,
            'scores': scores,
            'labels': labels
                }
            ]
        }
    return all_predictions

# def nms_on_multiple_prediocions(predictions, iou_threshold=0.5):
#     for tile_name, prediction in predictions.items():
#         boxes = prediction['image_coords'][0]['boxes']
#         scores = prediction['image_coords'][0]['scores']
#         labels = prediction['image_coords'][0]['labels']
#         keep = nms(boxes, scores, iou_threshold)
#         predictions[tile_name]['image_coords'][0]['boxes'] = boxes[keep]
#         predictions[tile_name]['image_coords'][0]['scores'] = scores[keep]
#         predictions[tile_name]['image_coords'][0]['labels'] = labels[keep]
#     return predictions

def coords_transform(tile_name, geometries, boxes, output_folder = 'temp'):
    """
    Transforms coordinates between image and geographical coordinates in the CRS stored in self.crs.
    
    Parameters:
    - tile_name: The name of the tile.
    - coords: A list of (x, y) tuples representing coordinates.
    - transform_type: The type of transformation ('image_to_geo' or 'geo_to_image').
    
    Returns:
    - transformed_coords: A list of (x, y) tuples representing transformed coordinates.
    """
    tile_name = os.path.basename(tile_name)
    tile_path = os.path.join(output_folder, tile_name)
    shapely_geometry = geometries[tile_name]
    bounds = shapely_geometry.bounds
    minx, miny, maxx, maxy = bounds
    tile_width = maxx - minx
    tile_height = maxy - miny
    with rasterio.open(tile_path) as src:
        image_height, image_width = src.height, src.width
    scale_x = tile_width / image_width
    scale_y = tile_height / image_height
    transformation_matrix = [scale_x, 0, 0, -scale_y, minx, maxy]
    transformed_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        points = [(xmin, ymin), (xmax, ymax)]
        #points = [(xmax, ymax), (xmin, ymin)]
        transformed_points = []
        for x, y in points:
            point = Point(x, y)
            transformed_point = affine_transform(point, transformation_matrix)
            transformed_points.append((transformed_point.x, transformed_point.y))
        transformed_boxes.append(transformed_points)

    return transformed_boxes

def create_shapefile(predictions, output_file_path, geometries, raster):
    with rasterio.open(raster) as src:
        crs = src.crs
    schema = {
        'geometry': 'Polygon',
        'properties': {'score': 'float', 
                        'tile_name': 'str',
                        }
    }
    
    with fiona.open(output_file_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as shp:
        for tile_name, pred in predictions.items():
            boxes = pred['image_coords'][0]['boxes'].cpu().numpy()  # Convert tensor to numpy array on CPU
            scores = pred['image_coords'][0]['scores']
            geo_coords = coords_transform(tile_name, geometries, boxes)
            for box, score in zip(geo_coords, scores):
                minx, miny = box[0]
                maxx, maxy = box[1]
                geom = shapely_box(minx, miny, maxx, maxy)
                shp.write({
                    'geometry': mapping(geom),
                    'properties': {'score': score.item(), 
                                   'tile_name': tile_name,
                                   }
                })

def plot_bboxes_on_images(predictions, output_folder):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import cm
    from matplotlib.colors import Normalize

    os.makedirs(output_folder, exist_ok=True)
    
    # Create a colormap and normalization
    cmap = cm.get_cmap('nipy_spectral')
    norm = Normalize(vmin=0, vmax=1)  # Assuming scores are between 0 and 1

    for tile_name, prediction in predictions.items():
        # Ensure the tile_name does not have double extension
        tile_name = tile_name.replace('.tif.tif', '.tif')
        
        # Find the corresponding image path
        image_path = prediction['tile_path'][0]
        if not os.path.exists(image_path):
            print(f"Image for {tile_name} not found.")
            continue
        
        # Load the image
        image = io.imread(image_path)
        
        # Create a figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 12))
        
        # Display the image
        ax.imshow(image)
        
        # Plot the bounding boxes
        for box, score in zip(prediction['image_coords'][0]['boxes'], prediction['image_coords'][0]['scores']):
            # Move the box coordinates to CPU and convert to numpy
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            color = cmap(norm(score.cpu().numpy()))
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        # Add color bar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Score')
        
        # Save the plot
        output_path = os.path.join(output_folder, f"{tile_name}.png")
        plt.savefig(output_path)
        print(f"Saved plot for {tile_name} to {output_path}")
        
        # Show the plot
        plt.show()
        
        # Close the figure
        plt.close(fig)

def main(config, checkpoint, output_file):

    print(f'Config Path: {config}')
    print(f'Checkpoint Path: {checkpoint}')
    print(f'Output File Path: {output_file}')

    cfg = load_config(config)
    raster = os.path.join(cfg['data']['path_to_dataset'], 
                            cfg['data']['dataset'], 
                            'orthomosaic.tif')
    tiles_shp = os.path.join(cfg['data']['path_to_dataset'],
                            cfg['data']['dataset'],
                            'tiles.shp')
    field_shp = os.path.join(cfg['data']['path_to_dataset'],
                            cfg['data']['dataset'],
                            'field_shape.shp')
    
    tile_paths, geometries = get_test_tiles(tiles_shp, raster, field_shp, output_file)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg['dataset']['image_size'], cfg['dataset']['image_size'])),
    ])

    dataset = CustomTileDataset(tile_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = load_model(cfg,checkpoint)

    print('Predicting bounding boxes...')
    predictions = predict_bounding_boxes(model, dataloader)

    print('Transforming image coordinates to crs coordinates...')
    create_shapefile(predictions, output_file, geometries, raster)
    print(f'Shapefile saved to {output_file}')
    # plot_bboxes_on_images(predictions, output_folder='test')
    # print(f'Plots saved to {'test'}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the experiments')
    parser.add_argument('-cfg','--config', type=str, help='The config file')
    parser.add_argument('-ckpt','--checkpoint', type=str, help='The checkpoint file')
    parser.add_argument('-out','--output_file', type=str, help='The output shapefile')
    args = parser.parse_args()
    config = args.config
    checkpoint = args.checkpoint
    output_file = args.output_file
    
    main(config,checkpoint,output_file)