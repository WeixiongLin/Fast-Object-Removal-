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
    width, height = mask_img.shape

    # count max width
    max_wid = 0
    for i in range(width):
        # initialize leftend and rightend of mask area as -1
        leftend = -1
        rightend = -1
        for j in range(height-1):
            if mask_img[i, j] > 30 and leftend == -1:
                leftend = j
            if mask_img[i, j] == 0 and mask_img[i-1, j] > 0 and i > 0:
                rightend = j
                break
        max_wid = max(max_wid, rightend-leftend)
    
    # print("max width: {}".format(max_wid))
    return max_wid
