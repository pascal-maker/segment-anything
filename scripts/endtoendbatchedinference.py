import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

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
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Load images
image1 = cv2.imread("fetus.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.imread('kidfetus.png')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Define image 1 boxes and image 2 boxes
image1_boxes = torch.tensor([
    [150, 50, 350, 300],
    [200, 100, 250, 200]
], device=sam.device)

image2_boxes = torch.tensor([
    [150, 50, 300, 200],
    [250, 100, 350, 150],
    [400, 50, 450, 100]
], device=sam.device)

# Resize transform
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

# Prepare batched input
def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

batched_input = [
     {
         'image': prepare_image(image1, resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]),
         'original_size': image1.shape[:2]
     },
     {
         'image': prepare_image(image2, resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
         'original_size': image2.shape[:2]
     }
]

# Run the model
batched_output = sam(batched_input, multimask_output=False)

# Visualize results
fig, ax = plt.subplots(1, 2, figsize=(20, 20))

ax[0].imshow(image1)
for mask in batched_output[0]['masks']:
    show_mask(mask.cpu().numpy(), ax[0], random_color=True)
for box in image1_boxes:
    show_box(box.cpu().numpy(), ax[0])
ax[0].axis('off')

ax[1].imshow(image2)
for mask in batched_output[1]['masks']:
    show_mask(mask.cpu().numpy(), ax[1], random_color=True)
for box in image2_boxes:
    show_box(box.cpu().numpy(), ax[1])
ax[1].axis('off')

plt.tight_layout()
plt.show()
