# mask 部分是亮的, 其他地方是暗的

import cv2
import numpy as np

img = cv2.imread("fall_mask.png", 0)
height, width = img.shape
mask = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        mask[i, j] = 255 - img[i, j]

cv2.imwrite("new_mask.png", mask)
