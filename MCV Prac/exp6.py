import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise to the image
gaussian_noise = np.random.normal(0, 25, image.shape)
noisy_image_gaussian = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)

# Add salt and pepper noise to the image
salt_pepper_ratio = 0.05
salt_pepper_noise = np.random.rand(*image.shape)
noisy_image_salt_pepper = image.copy()
noisy_image_salt_pepper[salt_pepper_noise < salt_pepper_ratio / 2] = 0
noisy_image_salt_pepper[salt_pepper_noise > 1 - salt_pepper_ratio / 2] = 255

# Apply mean filter
mean_filtered_gaussian = cv2.blur(noisy_image_gaussian, (5, 5))
mean_filtered_salt_pepper = cv2.blur(noisy_image_salt_pepper, (5, 5))

# Apply median filter
median_filtered_gaussian = cv2.medianBlur(noisy_image_gaussian, 5)
median_filtered_salt_pepper = cv2.medianBlur(noisy_image_salt_pepper, 5)

# Apply Gaussian filter
gaussian_filtered_gaussian = cv2.GaussianBlur(noisy_image_gaussian, (5, 5), 0)
gaussian_filtered_salt_pepper = cv2.GaussianBlur(
    noisy_image_salt_pepper, (5, 5), 0)

# Plot the images
plt.figure(figsize=(16, 12))

plt.subplot(3, 4, 1)
plt.imshow(noisy_image_gaussian, cmap='gray')
plt.title('Gaussian Noise')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(noisy_image_salt_pepper, cmap='gray')
plt.title('Salt and Pepper Noise')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(mean_filtered_gaussian, cmap='gray')
plt.title('Mean Filter (Gaussian Noise)')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(mean_filtered_salt_pepper, cmap='gray')
plt.title('Mean Filter (Salt & Pepper Noise)')
plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(median_filtered_gaussian, cmap='gray')
plt.title('Median Filter (Gaussian Noise)')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(median_filtered_salt_pepper, cmap='gray')
plt.title('Median Filter (Salt & Pepper Noise)')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(gaussian_filtered_gaussian, cmap='gray')
plt.title('Gaussian Filter (Gaussian Noise)')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(gaussian_filtered_salt_pepper, cmap='gray')
plt.title('Gaussian Filter (Salt & Pepper Noise)')
plt.axis('off')

plt.show()
