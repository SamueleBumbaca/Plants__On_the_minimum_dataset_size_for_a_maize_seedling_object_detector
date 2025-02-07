import numpy as np
from skimage import io, color
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage import graph
from skimage.measure import label
from skimage.color import rgb2hsv
from skimage.filters import sobel
from skimage.measure import regionprops
import matplotlib.pyplot as plt

def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def map_segment_colors(image, segments):
    out = np.zeros_like(image)
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        mean_color = image[mask].mean(axis=0)
        out[mask] = mean_color
    return out

def threshold_hsv(image_hsv,                             
                    green_hue_min=30,
                    green_hue_max=80,
                    green_saturation_min=50,
                    green_value_min=50,
                    green_saturation_max=255,
                    green_value_max=220):
    def convert_hsv_8bit(hsv):
        return hsv * 255
    image_hsv = convert_hsv_8bit(image_hsv)
    hue = image_hsv[:, :, 0]
    saturation = image_hsv[:, :, 1]
    value = image_hsv[:, :, 2]
    green_mask = (hue >= green_hue_min) & (hue <= green_hue_max) & \
                    (saturation >= green_saturation_min) & (saturation <= green_saturation_max) & \
                    (value >= green_value_min) & (value <= green_value_max)
    return green_mask

def process_image_threshold(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image / 255.0, 0, 1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    image_hsv = rgb2hsv(image)
    green_mask = threshold_hsv(image_hsv)
    labeled_patches = label(green_mask)
    num_pixels_per_patch = np.bincount(labeled_patches.ravel())
    return labeled_patches, num_pixels_per_patch

def process_image_felzenszwalb(image):
    segments_fz = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    rag = graph.rag_mean_color(image, segments_fz)
    merged_segments = graph.merge_hierarchical(segments_fz, rag, thresh=35, rag_copy=False,
                                               in_place_merge=True, merge_func=merge_mean_color,
                                               weight_func=_weight_mean_color)
    merged_image = map_segment_colors(image, merged_segments)
    image_hsv = rgb2hsv(merged_image)
    green_mask = threshold_hsv(image_hsv)
    labeled_patches = label(green_mask)
    num_pixels_per_patch = np.bincount(labeled_patches.ravel())
    return labeled_patches, num_pixels_per_patch

def process_image_slic(image):
    segments_slic = slic(image, n_segments=250, compactness=10, sigma=1)
    rag = graph.rag_mean_color(image, segments_slic)
    merged_segments = graph.merge_hierarchical(segments_slic, rag, thresh=35, rag_copy=False,
                                               in_place_merge=True, merge_func=merge_mean_color,
                                               weight_func=_weight_mean_color)
    merged_image = map_segment_colors(image, merged_segments)
    image_hsv = rgb2hsv(merged_image)
    green_mask = threshold_hsv(image_hsv)
    labeled_patches = label(green_mask)
    num_pixels_per_patch = np.bincount(labeled_patches.ravel())
    return labeled_patches, num_pixels_per_patch

def process_image_quickshift(image):
    segments_quick = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    rag = graph.rag_mean_color(image, segments_quick)
    merged_segments = graph.merge_hierarchical(segments_quick, rag, thresh=35, rag_copy=False,
                                               in_place_merge=True, merge_func=merge_mean_color,
                                               weight_func=_weight_mean_color)
    merged_image = map_segment_colors(image, merged_segments)
    image_hsv = rgb2hsv(merged_image)
    green_mask = threshold_hsv(image_hsv)
    labeled_patches = label(green_mask)
    num_pixels_per_patch = np.bincount(labeled_patches.ravel())
    return labeled_patches, num_pixels_per_patch

def process_image_watershed(image):
    gradient = sobel(color.rgb2gray(image))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    rag = graph.rag_mean_color(image, segments_watershed)
    merged_segments = graph.merge_hierarchical(segments_watershed, rag, thresh=35, rag_copy=False,
                                               in_place_merge=True, merge_func=merge_mean_color,
                                               weight_func=_weight_mean_color)
    merged_image = map_segment_colors(image, merged_segments)
    image_hsv = rgb2hsv(merged_image)
    green_mask = threshold_hsv(image_hsv)
    labeled_patches = label(green_mask)
    num_pixels_per_patch = np.bincount(labeled_patches.ravel())
    return labeled_patches, num_pixels_per_patch

def plot_labeled_patches(image_path):
    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Remove alpha channel if present
    methods = {
        'Threshold': process_image_threshold,
        'Felzenszwalb': process_image_felzenszwalb,
        'SLIC': process_image_slic,
        'Quickshift': process_image_quickshift,
        'Watershed': process_image_watershed
    }
    
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(25, 6))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for ax, (name, method) in zip(axes[1:], methods.items()):
        labeled_patches, num_pixels_per_patch = method(image)
        ax.imshow(labeled_patches, cmap='nipy_spectral')
        ax.set_title(name)
        ax.axis('off')
        
        # Annotate the number of pixels per patch
        for region in regionprops(labeled_patches):
            minr, minc, maxr, maxc = region.bbox
            ax.text(minc, minr, str(region.area), color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# Example usage
image_path = '/mnt/e/PhD/Paper2/dataset/base/SAGIT22_C_4175_2022_05_27-58-125.tif'
plot_labeled_patches(image_path)