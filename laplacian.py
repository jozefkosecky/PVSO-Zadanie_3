import cv2
import numpy as np

import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('labrador.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the Laplacian filter kernel
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

horizontal = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # s2
vertical = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # s1

# Prewitt Operator
h, w = gray_img.shape

# define images with 0s
newhorizontalImage = np.zeros((h, w))
newverticalImage = np.zeros((h, w))
newgradientImage = np.zeros((h, w))

# offset by 1
for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])

        newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

        sum = (-1 * gray_img[i + 1, j]) +\
        (-1 * gray_img[i - 1, j]) +\
        (-1 * gray_img[i, j + 1]) +\
        (-1 * gray_img[i, j - 1]) +\
        (-1 * gray_img[i - 1, j - 1]) +\
        (-1 * gray_img[i + 1, j + 1]) +\
        (-1 * gray_img[i - 1, j + 1]) +\
        (-1 * gray_img[i + 1, j - 1]) +\
        (8 * gray_img[i, j])

        newverticalImage[i - 1, j - 1] = abs(verticalGrad)

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        newgradientImage[i - 1, j - 1] = sum

plt.figure()
plt.title('result.png')
plt.imsave('result.png', newgradientImage, cmap='gray', format='png')
plt.imshow(newgradientImage, cmap='gray')
plt.show()

img = cv2.imread('result.png')
# Display the result
cv2.imshow('Laplacian', newgradientImage)
cv2.waitKey(0)
cv2.destroyAllWindows()