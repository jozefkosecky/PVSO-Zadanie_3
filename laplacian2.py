import numpy as np

import cv2
import numpy as np
import math

import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('labrador.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prewitt Operator
h, w = gray.shape

############### if you want use image, without blur uncomment this and comment other blur filters
# blur_image = gray.copy()

############################### Gaussian blur
blur_image = np.zeros((h, w), dtype=np.uint8)

gaussian_filter = np.array([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                            [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                            [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                            [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                            [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])

# Pad the image with zeros to avoid border issues
padded_image = np.pad(gray, ((2, 2), (2, 2)), mode='constant', constant_values=0)

# Apply the Gaussian filter
for i in range(2, h+2):
    for j in range(2, w+2):
        blur_image[i-2][j-2] = np.uint8(np.sum(padded_image[i-2:i+3, j-2:j+3] * gaussian_filter))


############################### Other filter for blur

# blur_image = np.zeros((h, w), dtype=np.uint8)
#
# for i in range(1, h-1):
#     for j in range(1, w-1):
#         blur_image[i][j] = np.uint8((1/9 * image[i-1][j-1]) +
#                                      (1/9 * image[i-1][j]) +
#                                      (1/9 * image[i-1][j+1]) +
#                                      (1/9 * image[i][j-1]) +
#                                      (1/9 * image[i][j]) +
#                                      (1/9 * image[i][j+1]) +
#                                      (1/9 * image[i+1][j-1]) +
#                                      (1/9 * image[i+1][j]) +
#                                      (1/9 * image[i+1][j+1]))



h, w = blur_image.shape
laplacian = np.zeros((h, w), dtype=np.uint8)
laplacian_diagonal = np.zeros((h, w), dtype=np.uint8)

for i in range(1, h-1):
    for j in range(1, w-1):
        laplacian_sum = -1 * blur_image[i + 1][j] + \
                        -1 * blur_image[i - 1][j] + \
                        -1 * blur_image[i][j + 1] + \
                        -1 * blur_image[i][j - 1] + \
                        (4 * blur_image[i][j])

        laplacian_diagonal_sum = (-1 * blur_image[i + 1][j]) + \
                                 (-1 * blur_image[i - 1][j]) + \
                                 (-1 * blur_image[i][j + 1]) + \
                                 (-1 * blur_image[i][j - 1]) + \
                                 (-1 * blur_image[i - 1][j - 1]) + \
                                 (-1 * blur_image[i + 1][j + 1]) + \
                                 (-1 * blur_image[i - 1][j + 1]) + \
                                 (-1 * blur_image[i + 1][j - 1]) + \
                                 (8 * blur_image[i][j])

        laplacian[i][j] = np.uint8(max(0, min(laplacian_sum, 255)))
        laplacian_diagonal[i][j] = np.uint8(max(0, min(laplacian_diagonal_sum, 255)))


# # Apply Gaussian blur with a kernel size of 5
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# blurred = gray.copy()
# Apply Laplacian operator with a kernel size of 3
laplacian_cv = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
# Convert the result to 8-bit unsigned integer
laplacian_cv = np.uint8(np.absolute(laplacian_cv))


# Display the result
cv2.imshow('Gray', gray)
cv2.imshow('Blur', blur_image)
cv2.imshow('laplacian_cv', laplacian_cv)
cv2.imshow('Laplacian', laplacian_diagonal)
cv2.imshow('Laplacian2', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

# blur_image = cv2.GaussianBlur(gray, (5, 5), 0)