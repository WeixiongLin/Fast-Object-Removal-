import os
import imageio


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def name_list(image_folder):
    dirpath = image_folder
    file_list = []
    for path, dirname, filename in os.walk(dirpath):
        file_list = filename

    file_list = [int(name[0:-4]) for name in filename]
    file_list.sort()
    filename = [str(name)+".png" for name in file_list]
    file_list = [os.path.join(image_folder, name) for name in filename]
    return file_list

def main(image_folder, gif_name):
    image_list = name_list(image_folder)
    duration = 0.35
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    outpath_of_gif = 'out.gif'
    main(image_folder="out/", gif_name=outpath_of_gif)
