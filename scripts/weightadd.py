import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from scipy.misc import imresize


# data_path = '/home/sconly/Documents/kitti_test'
data_path = '/node01_data5/monodepth2-test/analysis'

path1 = os.path.join(data_path, 'img')
path2 = os.path.join(data_path, 'std')


names1 = os.listdir(path1)
names1 = sorted(names1)
names2 = os.listdir(path2)
names2 = sorted(names2)

h = 256
w = 800

for i in range(len(names1)):
    img1 = plt.imread(os.path.join(path1, names1[i]))

    img2 = plt.imread(os.path.join(path2, names2[i]))

    img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    plt.imsave('/node01_data5/monodepth2-test/analysis/{}.png'.format(i), img.astype(np.uint8))

print('done!')
