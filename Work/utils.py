import cv2
import numpy as np

kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)


def calc_energy_map(img):
    b, g, r = cv2.split(img)
    b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
    g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
    r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
    return b_energy + g_energy + r_energy

def calc_neighbor_matrix(img, kernel):
    b, g, r = cv2.split(img)
    output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
             np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
             np.absolute(cv2.filter2D(r, -1, kernel=kernel))
    return output

# @author: Weixiong Lin
def branding(img, index, radius):
    x, y = index
    dx = [i for i in range(-radius, radius)]
    dy = [i for i in range(-radius, radius)]
    height, width = img.shape
    for i in dx:
        for j in dy:
            if x+i > 0 and x+i < height and y+j > 0 and y+j < width:
                img[x+i, y+j] = 255
    return img

# @author: Weixiong Lin
def max_width(mask):
    """Calculate maximum width of the mask area

    Scan row by row of the mask image.

    Args:
        mask: The path of mask image.
    
    Returns:
        max_width_mask: The maximum width of mask area.
    
    Raises:
        FileError: An error occured accessing the image object.
    """
    mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    # 旋转90
    mask_img = np.rot90(mask_img)
    cv2.imwrite("ro.jpg", mask_img)
    # 均值滤波
    # mask_img = cv2.blur(mask_img, (5, 5))
    # 二值化
    ret, mask_img = cv2.threshold(mask_img, 30, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("mask_img.png", mask_img)
    # print("image shape: {}".format(mask_img.shape))
    height, width = mask_img.shape

    # count max width
    max_wid = 0
    for i in range(height):
        # initialize leftend and rightend of mask area as -1
        leftend = -1
        rightend = -1
        for j in range(width-1):
            if mask_img[i, j] > 30 and leftend == -1:
                leftend = j
            if mask_img[i, j] == 0 and mask_img[i, j-1] > 0 and j > 0:
                rightend = j
                # cv2.imwrite("mask_img.png", branding(mask_img, (i, j), 2))
                # print("leftend:({}, {}); rightedn:({}, {})\n".format(i, leftend, i, rightend))
                break
        max_wid = max(max_wid, rightend-leftend)
    
    print("max width: {}".format(max_wid))
    return max_wid
