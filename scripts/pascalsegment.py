import sys
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2

# Function definitions for visualization
def show_mask(mask, ax, alpha=0.6, color=(0, 0.5, 1)):
    """Overlays a segmentation mask on an image in an axis."""
    h, w = mask.shape[:2]
    color_array = np.array(color)[:, None, None]  # Expand dimensions
    mask_image = (mask * color_array).transpose(1, 2, 0)  # Transpose to (height, width, channels)
    mask_image[:, :, 3] = alpha  # Set transparency
    ax.imshow(mask_image)



def show_points(coords, labels, ax, marker_size=250, label_colors={'0': 'red', '1': 'green'}):
    """Plots points on an image axis based on labels and colors."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=label_colors['1'], marker='o', s=marker_size, edgecolor='white', linewidth=1.5)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color=label_colors['0'], marker='x', s=marker_size, edgecolor='white', linewidth=1.5)

def show_box(box, ax, color=(0, 1, 0), thickness=2):
    """Visualizes a bounding box on an image axis."""
    x1, y1, x2, y2 = box
    plt.gca().add_patch(plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1,
                                      facecolor='none', edgecolor=color, linewidth=thickness))

# Set paths and device
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

# Load the SAM model and predictors
sys.path.append("..")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Load the image
image_path = "fetus.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set the image embedding
predictor.set_image(image)

# Define the bounding box based on coordinates from the second script
input_box = np.array([50, 50, 250, 150])  # Adjusted coordinates

# Optional: Visualize the bounding box
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_box(input_box, plt.gca())
plt.title("Input Bounding Box")
plt.axis('off')
plt.show()

# Make predictions using the bounding box
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],  # Provide the bounding box
    multimask_output=False,  # Not using multimask in this example
)

# Visualize the predicted mask and bounding box
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.title("Segmented Object")
plt.show()
