import matplotlib.pyplot as plt
import os
from scipy.misc import imresize
import numpy as np



def montage_image(folder_list, dst_path, h=256, w=800, row=2, col=2):
    names_list = []
    for i in range(len(folder_list)):
        names = os.listdir(folder_list[i])
        names = sorted(names)
        names_list.append(names)

    black = 10
    for i in range(len(names_list[0])):
        BigImage = np.ones((h * row + black * (row-1), w * col + black * (col-1), 3))*255
        for r in range(row):
            for c in range(col):
                idx = r*col+c
                img = plt.imread(os.path.join(folder_list[idx], names_list[idx][i]))
                print(os.path.join(folder_list[idx], names_list[idx][i]))
                img = imresize(img[:,:,:3], (h, w, 3))
                BigImage[r*(h+black):r*(h+black)+h, c*(w+black):c*(w+black)+w,:] =img[:,:,:3]
        plt.imsave(dst_path+'/{}.jpg'.format(i), BigImage.astype(np.uint8))
    print('done!')


if __name__=="__main__":
    data_path = '/node01_data5/monodepth2-test/analysis/head'
    folder_list = []
    folder_list.append(os.path.join(data_path, 'img'))
    folder_list.append(os.path.join(data_path, 'depth'))
    folder_list.append(os.path.join(data_path, 'ori_loss'))
    folder_list.append(os.path.join(data_path, 'loss'))
    dst_path = '/node01_data5/monodepth2-test/analysis/head/montage'
    montage_image(folder_list, dst_path, 256, 800, 2, 2)

    # data_path = '/node01_data5/monodepth2-test/kitti_test'
    # folder_list = []
    # folder_list.append(os.path.join(data_path, 'img'))
    # folder_list.append(os.path.join(data_path, 'mono+stereo_1024_320'))
    # folder_list.append(os.path.join(data_path, 'ms_baseline_50_256_800_0901'))
    # dst_path = '/node01_data5/monodepth2-test/kitti_test/montage'
    # montage_image(folder_list, dst_path, 256, 800, 3, 1)