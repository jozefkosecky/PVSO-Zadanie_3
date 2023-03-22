import numpy as np

import cv2
import numpy as np

import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('labrador.jpg')

# Convert the image to grayscale
gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prewitt Operator
h, w = gr.shape

output = np.zeros((h, w), dtype=np.uint8)
output3 = np.zeros((h, w), dtype=np.uint8)

for i in range(1, h-1):
    for j in range(1, w-1):
        sum2 = 1 * gr[i+1][j] + \
               1 * gr[i-1][j] + \
               1 * gr[i][j+1] + \
               1 * gr[i][j-1] + \
               (-4 * gr[i][j])

        sum = (-1 * gr[i+1][j]) + \
              (-1 * gr[i-1][j]) + \
              (-1 * gr[i][j+1]) + \
              (-1 * gr[i][j-1]) + \
              (-1 * gr[i-1][j-1]) + \
              (-1 * gr[i+1][j+1]) + \
              (-1 * gr[i-1][j+1]) + \
              (-1 * gr[i+1][j-1]) + \
              (8 * gr[i][j])
        output3[i][j] = np.uint8(max(0, min(sum, 255)))
        output[i][j] = np.uint8(max(0, min(sum2, 255)))


# Display the result
cv2.imshow('Laplacian', output3)
cv2.imshow('Laplacian2', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
