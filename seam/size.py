import cv2

img1 = cv2.imread("out/1.png", 0)
img2 = cv2.imread("out/115.png", 0)
img3 = cv2.imread("../figures/duck.jpg")
img4 = cv2.imread("../Work/deleted.png")

print(img1.shape)
print(img2.shape)
print(img3.shape)
print(img4.shape)

# cv2.imwrite("size.jpg", img3[:-1, :, :])
