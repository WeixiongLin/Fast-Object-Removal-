import os
import imageio
import numpy as np
import cv2


# 用 imageio 制作 gif 图片
def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

# 返回 filename list
def name_list(image_folder, rotate):
    dirpath = image_folder
    file_list = []
    for path, dirname, filename in os.walk(dirpath):
        file_list = filename
    # rotate
    if rotate:
        for filename in file_list:
            print(filename)
            img = cv2.imread("out/"+filename)
            img_ro = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite("out_ro/" + filename, img_ro)

    file_list = [int(name[0:-4]) for name in file_list]
    file_list.sort()
    print(file_list)
    filename = [str(name)+".png" for name in file_list]
    file_list = [os.path.join(image_folder, name) for name in filename]
    return file_list

def main(image_folder, gif_name):
    image_list = name_list(image_folder, rotate=False)
    duration = 0.35
    new_image_list = []
    for i in range(len(image_list)):
        if i % 20 == 0:
            new_image_list.append(image_list[i])
    create_gif(new_image_list, gif_name, duration)


if __name__ == '__main__':
    outpath_of_gif = 'fall_out_small.gif'
    main(image_folder="out_ro/", gif_name=outpath_of_gif)
    # img1 = cv2.imread("../figures/fall.jpg")
    # print(img1.shape)
    # img2 = cv2.imread("out/133.png")
    # print(img2.shape)