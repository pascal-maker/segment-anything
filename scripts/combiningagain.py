import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2

# Function definitions for visualization
def show_mask(mask, ax, random_color=False):
    """Overlays a segmentation mask on an image in an axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Plots points on an image axis based on labels."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Plots a bounding box on an image axis."""
    x_min, y_min, x_max, y_max = box
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)

# Set paths and device (adjust as needed)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Assuming you have the correct checkpoint file
model_type = "vit_h"
device = "cpu"  # Use "gpu" if available

# Load the SAM model and predictors
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Load your image (replace with your path)
image = cv2.imread("fetus.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set the image embedding using predictor.set_image
predictor.set_image(image)

# Input data (adjusted based on the second script)
input_point = np.array([[150, 100]])  # Adjusted based on the coordinates of the second script
input_label = np.array([1])  # Label 1 for foreground point
input_box = np.array([50, 50, 250, 150])  # Adjusted based on the coordinates of the second script

# Make predictions
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)

# Visualize results
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
