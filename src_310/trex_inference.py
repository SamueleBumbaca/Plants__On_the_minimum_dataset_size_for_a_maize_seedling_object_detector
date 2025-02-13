from trex import TRex2APIWrapper, visualize
import torch
from torchvision.ops import nms
from PIL import Image
import numpy as np
import argparse
import os

# Initialize the client with your API token.
with open("src_310/api_token.txt", "r") as f:
    token = f.read().strip()  # Your API Token Here

# Get images paths

# Convert the images to png format
# Get the bounding boxes
# Initialize the client with your API token.
trex2 = TRex2APIWrapper(token)
# Define the prompts and the corresponding bounding boxes.
emb_prompts = [
        {
            "prompt_image": prompt_image,
            "rects": rect,
        } for prompt_image, rect in zip(train_images, rects)
    ]
# Customize the embedding.
embedding_url = trex2.customize_embedding(emb_prompts)
print(f"Customized embedding URL: {embedding_url}")#TODO:save the embedding_url to a file
# Define the prompts for inference.
inf_prompts = [
    {
        "image": image,
        "prompts": [
            {"category_id": 1, "embd": embedding_url},
        ],
    } for image in eval_images
]
# Perform inference.
results = trex2.embedding_inference(inf_prompts)
# # filter out the boxes with low score
# filtered_results = []
# for result in results:
#     scores = np.array(result["scores"])
#     labels = np.array(result["labels"])
#     boxes = np.array(result["boxes"])
#     filter_mask = scores > args.box_threshold
#     filtered_result = {
#         "scores": scores[filter_mask],
#         "labels": labels[filter_mask],
#         "boxes": boxes[filter_mask],
#     }
#     filtered_results.append(filtered_result)
#TODO: instead of filter out the boxes with low score apply NMS