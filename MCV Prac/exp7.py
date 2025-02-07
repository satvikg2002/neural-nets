import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Fourier transform
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# Create a mask to filter out high frequencies
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
r = 80  # adjust the radius to filter out different frequency components
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0

# Apply the mask to the frequency domain representation
f_shift_filtered = f_shift * mask

# Inverse Fourier transform to get the image back
f_ishift = np.fft.ifftshift(f_shift_filtered)
filtered_image = np.fft.ifft2(f_ishift)
filtered_image = np.abs(filtered_image)

# Display original and filtered images
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
