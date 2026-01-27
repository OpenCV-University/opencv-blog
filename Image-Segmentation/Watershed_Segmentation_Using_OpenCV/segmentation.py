import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, title="", ax=None):
    if ax is None:
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis("off")
    else:
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis("off")

# Load image
img = cv2.imread("coins.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold 
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Noise removal 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background 
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Distance transform 
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling 
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Watershed 
markers_ws = cv2.watershed(img, markers)
label_ids = np.unique(markers_ws)
label_ids = label_ids[label_ids > 1]  # remove background & boundary

colored_labels = np.zeros_like(img_rgb)

rng = np.random.default_rng(42)

for label in label_ids:
    color = rng.integers(50, 255, size=3)
    colored_labels[markers_ws == label] = color

# Overlay colors on original image
alpha = 0.6
overlay = img_rgb.copy()
mask = markers_ws > 1

overlay[mask] = (
    alpha * colored_labels[mask] +
    (1 - alpha) * img_rgb[mask]
).astype(np.uint8)

# Draw watershed boundaries in red
overlay[markers_ws == -1] = [255, 0, 0]
segmented = img_rgb.copy()
segmented[markers_ws == -1] = [255, 0, 0]  # boundaries in red


fig, ax = plt.subplots(2, 3, figsize=(15, 10))

imshow(gray, "Gray", ax[0,0])
imshow(binary, "Binary", ax[0,1])
imshow(opening, "After Morphology", ax[0,2])

imshow(dist, "Distance Transform", ax[1,0])
imshow(markers_ws, "Watershed Labels", ax[1,1])
ax[1,2].imshow(segmented)
ax[1,2].imshow(overlay)
ax[1,2].set_title("Colored Watershed Segmentation")
ax[1,2].axis("off")
ax[1,2].axis("off")

plt.tight_layout()
plt.show()

labels = np.unique(markers_ws)
coin_count = len(labels[labels > 1])
print("Number of coins:", coin_count)