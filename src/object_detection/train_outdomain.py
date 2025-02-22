from object_detection.dataset.field_data_module import FieldDataModule
from object_detection.models import DETR, YOLO
import torch
import yaml
import argparse
import subprocess
import shutil
import glob
import random
from os import listdir, makedirs
from os.path import join, exists, basename

# Function to copy files from one folder to another
def copy_files(file_list, dst_folder):
    for src_file in file_list:
        file_basename = basename(src_file)
        dst_file = join(dst_folder, file_basename)
        shutil.copy(src_file, dst_file)

def split_dataset(images, train_size):
    # Randomly shuffle the images
    random.shuffle(images)
    # Split the images into train and val sets
    print(f"Total number of images: {len(images)}")
    train_images = images[:int(train_size * len(images))]
    print(f"Training set size: {len(train_images)}")
    val_images = images[int(train_size * len(images)):]
    print(f"Validation set size: {len(val_images)}")
    return train_images, val_images

def images2dset(images):
    labels = []
    for image in images:
        label = image.replace('.tif', '.txt')
        labels.append(label)
    images.extend(labels)
    return images

# Function to write out-domain dataset
def write_out_domain_data(dataset, yolo_dataset_path, train_size):
    one_class_yaml = """
                    names:
                    - object
                    nc: 1
                    train: train
                    val: val
                    """
    out_domain_path = join('dataset','out_domain')
    out_domain_dataset_path = join(out_domain_path, dataset, 'yolo_dataset')
    if not exists(yolo_dataset_path):
        makedirs(yolo_dataset_path)
    # List all the images in the out-domain dataset
    images = glob.glob(join(out_domain_dataset_path, '*.tif'))
    train_images, val_images = split_dataset(images, train_size)
    train_set = images2dset(train_images)
    val_set = images2dset(val_images)
    # Create the train and val folders
    train_folder = join(yolo_dataset_path, 'train')
    val_folder = join(yolo_dataset_path, 'val')
    for folder in [train_folder, val_folder]:
        makedirs(folder, exist_ok=True)
    # Copy train and val files from the out-domain dataset
    copy_files(train_set, train_folder)
    copy_files(val_set, val_folder)
    # Save the dataset.yaml file in the new dataset folder
    with open(join(yolo_dataset_path, 'dataset.yaml'), 'w') as file:
        yaml.safe_dump(yaml.safe_load(one_class_yaml), file, default_flow_style=False)

# Function to merge the in-domain and out-domain data
def merge_in_and_out_datasets(yolo_dataset_path, 
                              train_split, 
                              out_domain_path = f'dataset/out_domain/AllDatasets/yolo_dataset'):
    
    out_domain_images = glob.glob(join(out_domain_path, '*.tif'))
    out_domain_train_images, out_domain_val_images = split_dataset(out_domain_images, train_split)
    out_domain_train_set = images2dset(out_domain_train_images)
    out_domain_val_set = images2dset(out_domain_val_images)
    # Copy in the train and val folders
    val_folder = join(yolo_dataset_path, 'val')
    train_folder = join(yolo_dataset_path, 'train')
    copy_files(out_domain_train_set, train_folder)
    copy_files(out_domain_val_set, val_folder)

# Set the default precision for matrix multiplication to float32
torch.set_float32_matmul_precision('medium')
# Main function to train the model
def main(config):
    with open(config, 'r') as file:
        cfg= yaml.safe_load(file)
    # Set the seed for reproducibility
    seed = cfg['train']['seed']
    torch.manual_seed(seed)
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get the git commit version
    try:
        git_commit_version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        git_commit_version = 'unknown'
    cfg['git_commit_version'] = git_commit_version
    if cfg['data']['dataset'].isdigit(): # If the dataset is an integer, it is an in-domain dataset
        # Initialize the data module
        data = FieldDataModule(config)
        cfg['data']['yolo_dataset'] = data.convert_to_yolo_format()
        with open(config, 'r') as file:
            cfg= yaml.safe_load(file)
        merge_in_and_out_datasets(cfg['data']['yolo_dataset'],
                                  float(cfg['data']['train_split'])
                                  )
    else:
        write_out_domain_data(cfg['data']['dataset'], 
                              cfg['data']['yolo_dataset'], 
                              float(cfg['data']['train_split']))
    # Load the model type from the config file
    model_type = cfg['train']['model']
    # Load the model class based on the model type
    if 'YOLO' in model_type.upper():
        print("Using YOLO model")
        model = YOLO.YOLO_model(cfg)
    elif 'DETR' in model_type.upper():
        print("Using DETR model")
        model = DETR.DETR_model(cfg)
    else:
        raise NotImplementedError

    model.train_model()
    print("Training completed")

# Entry point of the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with the given configuration.')
    parser.add_argument('-cfg', '--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)