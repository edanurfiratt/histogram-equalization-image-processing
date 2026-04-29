import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image file not found. Please check input.jpg")
    exit()

# Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Save output
cv2.imwrite("output.png", equalized_image)

# Show original and result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap="gray")
plt.title("Histogram Equalized")
plt.axis("off")

plt.tight_layout()
plt.show()