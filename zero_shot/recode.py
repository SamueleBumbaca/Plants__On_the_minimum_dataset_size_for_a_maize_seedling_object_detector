import json

# Load the JSON data
with open('/home/samuelebumbaca/repositories/Zero_shot_detection/datasets/2/annotations/test.json', 'r') as file:
    data = json.load(file)

# Create a mapping from file_name to new ID
file_name_to_id = {}
new_id = 1

# Create a mapping from old image_id to new image_id
old_to_new_id = {}

# First, identify all unique file_names and assign new IDs
for i, image in enumerate(data['images']):
    if 'file_name' in image:
        file_name = image['file_name']
        if file_name not in file_name_to_id:
            file_name_to_id[file_name] = new_id
            new_id += 1
            
# Create a mapping from old image IDs to new IDs and prepare unique images
unique_images = {}
for image in data['images']:
    if 'file_name' in image and 'id' in image:
        old_id = image['id']
        new_id = file_name_to_id[image['file_name']]
        
        # Store the mapping from old to new ID
        old_to_new_id[old_id] = new_id
        
        # Update image ID
        image['id'] = new_id
        
        # Store only one copy of each unique image based on file_name
        unique_images[image['file_name']] = image

# Replace the images list with the unique images
data['images'] = list(unique_images.values())

# Update the image_id in the annotations section
for annotation in data['annotations']:
    if 'image_id' in annotation and annotation['image_id'] in old_to_new_id:
        annotation['image_id'] = old_to_new_id[annotation['image_id']]

# Check for duplicate annotations
bbox_map = {}
duplicate_count = 0
for anno in data['annotations']:
    if 'image_id' in anno and 'bbox' in anno and isinstance(anno['bbox'], list) and len(anno['bbox']) == 4:
        key = (anno['image_id'], tuple(anno['bbox']))
        if key in bbox_map:
            duplicate_count += 1
            print(f"Duplicate bbox found for image_id {anno['image_id']}, bbox {anno['bbox']}")
        bbox_map[key] = anno.get('id')

# Filter out incomplete annotations
valid_annotations = []
for anno in data['annotations']:
    if ('id' in anno and 'image_id' in anno and 'category_id' in anno and 
        'bbox' in anno and isinstance(anno['bbox'], list) and len(anno['bbox']) == 4):
        valid_annotations.append(anno)

# Replace annotations with valid ones
data['annotations'] = valid_annotations

# Save the updated JSON data
with open('/home/samuelebumbaca/repositories/Zero_shot_detection/datasets/1/annotations/test.json', 'w') as file:
    json.dump(data, file, indent=4)

def fix_annotations(json_file):
    # Load the JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Check for duplicate annotations
    bbox_map = {}
    duplicate_count = 0
    for anno in data['annotations']:
        if 'image_id' in anno and 'bbox' in anno and isinstance(anno['bbox'], list) and len(anno['bbox']) == 4:
            key = (anno['image_id'], tuple(anno['bbox']))
            if key in bbox_map:
                duplicate_count += 1
                print(f"Duplicate bbox found for image_id {anno['image_id']}, bbox {anno['bbox']}")
            bbox_map[key] = anno.get('id')
    
    # Filter out incomplete annotations
    valid_annotations = []
    for anno in data['annotations']:
        if ('id' in anno and 'image_id' in anno and 'category_id' in anno and 
            'bbox' in anno and isinstance(anno['bbox'], list) and len(anno['bbox']) == 4):
            valid_annotations.append(anno)
    
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Valid annotations: {len(valid_annotations)}")
    print(f"Removed incomplete annotations: {len(data['annotations']) - len(valid_annotations)}")
    print(f"Found {duplicate_count} duplicate annotations")
    
    # Replace annotations with valid ones
    data['annotations'] = valid_annotations
    
    # Save the cleaned data
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)