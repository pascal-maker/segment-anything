import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch

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

def show_box(box, ax):
    """Plots a bounding box on an image axis."""
    x_min, y_min, x_max, y_max = box
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)

# Set paths and device (adjust as needed)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Assuming you have the correct checkpoint file
model_type = "vit_h"
device = "cpu"  # Use "gpu" if available

# Load the SAM model and predictor
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Load your image (replace with your path)
image = cv2.imread("fetus.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set the image embedding
predictor.set_image(image)

# Batched prompt inputs
input_boxes = torch.tensor([
    [50, 50, 250, 150],
    [150, 100, 300, 200]
], device=predictor.device)

# Transform the boxes to the input frame
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

# Make predictions using predict_torch
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# Visualize results
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask, box in zip(masks, input_boxes):
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()
