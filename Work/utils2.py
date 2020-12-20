'''usage:
delete_seams(img,num_paths)
img is np array'''

import numpy as np
import cv2


def calc_energy_map(out_image):
    b, g, r = cv2.split(out_image)
    b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
    g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
    r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
    return b_energy + g_energy + r_energy

def cumulative_map_backward(energy_map):
    m, n = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, m):
        for col in range(n):
            output[row, col] = \
                energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
    return output


def find_seam(cumulative_map):
    m, n = cumulative_map.shape
    output = np.zeros((m,), dtype=np.uint32)
    output[-1] = np.argmin(cumulative_map[-1])
    for row in range(m - 2, -1, -1):
        prv_x = output[row + 1]
        if prv_x == 0:
            output[row] = np.argmin(cumulative_map[row, : 2])
        else:
            output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
    return output


def delete_seam(out_image, seam_idx):
    m, n = out_image.shape[: 2]
    output = np.zeros((m, n - 1, 3))
    for row in range(m):
        col = seam_idx[row]
        output[row, :, 0] = np.delete(out_image[row, :, 0], [col])
        output[row, :, 1] = np.delete(out_image[row, :, 1], [col])
        output[row, :, 2] = np.delete(out_image[row, :, 2], [col])
    out_image = np.copy(output)
    return out_image


def add_seam(out_image, seam_idx):
    m, n = out_image.shape[: 2]
    output = np.zeros((m, n + 1, 3))
    for row in range(m):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(out_image[row, col: col + 2, ch])
                output[row, col, ch] = out_image[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = out_image[row, col:, ch]
            else:
                p = np.average(out_image[row, col - 1: col + 1, ch])
                output[row, : col, ch] = out_image[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = out_image[row, col:, ch]
    out_image = np.copy(output)
    return out_image


def update_seams(remaining_seams, current_seam):
    output = []
    for seam in remaining_seams:
        seam[np.where(seam >= current_seam)] += 2
        output.append(seam)
    return output

def seams_insertion(out_image, num_pixel):
    '''delete vertical paths'''
    temp_image = np.copy(out_image)
    seams_record = []

    for dummy in range(num_pixel):
        energy_map = calc_energy_map(out_image).astype(np.int32)
        cumulative_map = cumulative_map_backward(energy_map)
        seam_idx = find_seam(cumulative_map)
        seams_record.append(seam_idx)
        out_image = delete_seam(out_image, seam_idx)

    out_image = np.copy(temp_image)
    n = len(seams_record)
    for dummy in range(n):
        seam = seams_record.pop(0)
        out_image = add_seam(out_image, seam)
        seams_record = update_seams(seams_record, seam)
    return out_image

if __name__=='__main__':
    img_path='deleted.png'
    img=cv2.imread(img_path)
    inserted = seams_insertion(img,60)
    cv2.imwrite('inserted.jpg',inserted)