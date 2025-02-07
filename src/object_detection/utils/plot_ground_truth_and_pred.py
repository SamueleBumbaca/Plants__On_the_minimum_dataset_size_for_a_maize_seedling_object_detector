import yaml
import torch
import fiona
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, Point
from shapely.affinity import affine_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from object_detection.models.faster_RCNN_FPN import MyFasterRCNN
import click
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize
from skimage import io
from object_detection.dataset.field_data_module import FieldDataModule

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(cfg, checkpoint_path):
    model = MyFasterRCNN(cfg, cfg['dataset']['train_mean'], cfg['dataset']['train_std'])
    model.load_checkpoint(checkpoint_path)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

def get_test_tiles(shp_path, raster_path, output_folder='temp'):
    os.makedirs(output_folder, exist_ok=True)
    tile_names = []
    with fiona.open(shp_path, 'r') as shapefile:
        shp_crs = shapefile.crs
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            if shp_crs != raster_crs:
                raise ValueError("CRS of the shapefile and raster do not match")
            new_raster_paths = []
            geometries = {}
            for feature in shapefile:
                polygon = shape(feature['geometry'])
                name = feature['properties']['NAME']
                out_image, out_transform = mask(src, [polygon], crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                output_path = os.path.join(output_folder, name)
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                new_raster_paths.append(output_path)
                geometries[name] = polygon
                tile_names.append(name)
    return tile_names, new_raster_paths, geometries

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
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    all_predictions = []
    for images, path in dataloader:
        tile_name = os.path.basename(path[0])
        images = [img.to(device) for img in images]
        with torch.no_grad():
            predictions = model(images)
        all_predictions.append({'tile_name': tile_name, 
                                'tile_path': path, 
                                'image_coords': predictions})
    return all_predictions

def coords_transform(tile_name, geometries, boxes, transform_type='image_to_geo', output_folder='temp'):
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
    match transform_type:
        case 'image_to_geo':
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                transformed_points = []
                for x, y in points:
                    point = Point(x, y)
                    transformed_point = affine_transform(point, transformation_matrix)
                    transformed_points.append((transformed_point.x, transformed_point.y))
                transformed_boxes.append(transformed_points)
        case 'geo_to_image':
            inverse_transformation_matrix = [1/scale_x, 0, 0, -1/scale_y, -minx/scale_x, maxy/scale_y]
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                transformed_points = []
                for x, y in points:
                    point = Point(x, y)
                    transformed_point = affine_transform(point, inverse_transformation_matrix)
                    transformed_points.append((transformed_point.x, transformed_point.y))
                transformed_boxes.append(transformed_points)
        case _:
            raise ValueError("Invalid transform_type. Use 'image_to_geo' or 'geo_to_image'.")

    return torch.tensor(transformed_boxes)

def plot_bounding_boxes(dataset, tile_names, cfg, predictions):
    if dataset == 'train':
        stage = 'fit'
    elif dataset == 'val':
        stage = 'fit'
    elif dataset == 'test':
        stage = 'test'
    else:
        raise ValueError("Dataset must be 'train', 'val', or 'test'")

    data_module = FieldDataModule(cfg)
    data_module.setup(stage=stage)

    if dataset == 'train':
        data = data_module.train_dataloader()
    elif dataset == 'val':
        data = data_module.val_dataloader()
    elif dataset == 'test':
        data = data_module.test_dataloader()
    else:
        raise ValueError("Dataset must be 'train', 'val', or 'test'")

    cmap = cm.get_cmap('nipy_spectral')
    norm = Normalize(vmin=0, vmax=1)

    for id,tile_name_gt in enumerate(tile_names):
        if dataset == 'train':
            idx = data_module.train_dataset.tile_names.index(tile_name_gt)
        elif dataset == 'val':
            idx = data_module.val_dataset.tile_names.index(tile_name_gt)
        elif dataset == 'test':
            idx = data_module.test_dataset.tile_names.index(tile_name_gt)
        img, target = data.dataset[idx]
        #img = transforms.ToTensor()(img)  # Convert image to tensor
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0))
        print(f"ID: {idx}")
        boxes_gt = []
        for box_gt in target['boxes']:
            x_min, y_min, x_max, y_max = box_gt
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            #print(f"Ground truth bounding box: {box}")
            boxes_gt.append([x_min, y_min, x_max, y_max])

        boxes_pred = []
        tile_name_pred = os.path.basename(predictions[id]['tile_path'][0])
        for box_pred, score in zip(predictions[id]['image_coords'][0]['boxes'], predictions[id]['image_coords'][0]['scores']):
            xmin, ymin, xmax, ymax = box_pred.cpu().numpy()
            color = cmap(norm(score.cpu().numpy()))
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            #print(f"Predicted {tile_name_gt} {tile_name_pred} bounding box: {box}, Score: {score}")
            boxes_pred.append([xmin, ymin, xmax, ymax])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Score')
        plt.show()

        # Compute the IoU between the ground truth and predicted bounding boxes
        ious = []
        for box_gt in boxes_gt:
            for box_pred in boxes_pred:
                iou = compute_iou(box_gt, box_pred)
                if iou > 0:
                    ious.append(iou)
        print(f"Mean IoU: {sum(ious)/len(ious)}")

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters
    ----------
    box1 : list or array
        [xmin, ymin, xmax, ymax]
    box2 : list or array
        [xmin, ymin, xmax, ymax]
    
    Returns
    -------
    float
        IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    # Compute the area of intersection rectangle
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

@click.command()
@click.option('-cfg', '--config', required=True, type=str, help='Path to the config file')
@click.option('-ckpt', '--checkpoint', required=True, type=str, help='Path to the checkpoint file')
@click.option('-out', '--output_file', required=True, type=str, help='Path to the output shapefile')
def main(config, checkpoint, output_file):
    config_path = config
    checkpoint_path = checkpoint
    output_file_path = output_file

    print(f'Config Path: {config_path}')
    print(f'Checkpoint Path: {checkpoint_path}')
    print(f'Output File Path: {output_file_path}')

    cfg = load_config(config_path)

    trial_date = cfg['dataset']['trial'] + '_' + cfg['dataset']['date']
    raster = os.path.join(cfg['dataset']['path_to_dataset'], trial_date, 'orthomosaic.tif')

    model = load_model(cfg, checkpoint_path)
    tile_names, tile_paths, geometries = get_test_tiles('temp/val_tiles.shp', raster)
    print('Number of tiles:', len(tile_names))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg['dataset']['image_size'], cfg['dataset']['image_size'])),
    ])

    dataset = CustomTileDataset(tile_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print('Predicting bounding boxes...')
    predictions = predict_bounding_boxes(model, dataloader)
    print('Transforming image coordinates to crs coordinates...')
    plot_bounding_boxes('val', tile_names, config_path, predictions)

if __name__ == '__main__':
    main()