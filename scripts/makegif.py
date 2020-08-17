import imageio
import os

def create_gif(img_dir, image_list, gif_name, duration=0.35):
    frames = []
    i=0
    for image_name in image_list:
        i=i+1
        print(i)
        frames.append(imageio.imread(img_dir+image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    img_dir='/home/sconly/Documents/eccv/res/'
    image_list=sorted(os.listdir(img_dir))
    gif_name = '/home/sconly/Documents/eccv/res.gif'
    duration = 0.1
    create_gif(img_dir, image_list, gif_name, duration)


if __name__ == '__main__':
    main()