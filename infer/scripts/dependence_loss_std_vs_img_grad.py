from __future__ import absolute_import, division, print_function

import os
import cv2
import argparse
import numpy as np
from mmcv import Config
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.datasets.kitti_dataset import KITTIRAWDataset
from mono.datasets.utils import readlines
from mono.model.registry import MONO

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


"""
outputs[("axisangle", 0, f_i)] = axisangle
outputs[("translation", 0, f_i)] = translation
outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
outputs[("disp", scale)]
outputs[("depth", 0, scale)]
outputs[("sample", frame_id, scale)] = pix_coords
outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, 0)], outputs[("sample", frame_id, scale)], padding_mode="border")
outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, 0)]
"""


def gradient(D):
    h,w,c = D.shape
    dy = np.abs(D[1:, :] - D[:-1,:])
    dx = np.abs(D[:, 1:] - D[:, :-1])
    dx = np.resize(dx, (h, w, c))
    dy = np.resize(dy, (h, w, c))
    return dx, dy


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--data_path',
                        default='/node01_data5/kitti_raw')
    parser.add_argument('--output_path',
                        default='/node01_data5/monodepth2-test/analysis/ms_baseline')
    parser.add_argument('--model_path',
                        default='/node01_data5/monodepth2-test/model/ms_baseline/ms_baseline_50_256_800_0901.pth')

    parser.add_argument('--out_path',
                        help='needed by job client')
    parser.add_argument('--in_path',
                        help='needed by job client')
    parser.add_argument('--pretrained_path',
                        help='needed by job client')
    parser.add_argument('--job_name',
                        help='needed by job client')
    parser.add_argument('--job_id',
                        help='needed by job client')

    parser.add_argument('--height',
                        default=256)
    parser.add_argument('--width',
                        default=800)
    args = parser.parse_args()
    return args


def visualize(opt):
    device = [0]
    opt.gpu_num = len(device)

    filenames = readlines("mono/datasets/splits/exp/val_files.txt")

    dataset = KITTIRAWDataset(opt.data_path,
                              filenames,
                              opt.height,
                              opt.width,
                              [0, -1, 1],
                              is_train=True,
                              gt_depth_path='/node01_data5/monodepth2-test/monodepth2/gt_depths.npz')

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    cfg = Config.fromfile('./config/cfg_kitti_uncert.py')
    cfg.model['imgs_per_gpu'] = 1
    model_name = cfg.model['name']
    model = MONO.module_dict[model_name](cfg.model)

    checkpoint = torch.load(opt.model_path)

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            losses = []
            interval = 10
            for k in range(-interval, 1+interval):
                outputs, loss_dict = model(inputs, k/100+1)
                loss = outputs[('min_reconstruct_loss', 0)].cpu().numpy()#[1,h,w]
                losses.append(loss)

            losses = np.concatenate(losses, axis=0)
            std_loss = np.std(losses, axis=0) #[h,w]

            target = inputs[("color", 0, 0)].squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()
            dx, dy = gradient(target)
            dx = np.linalg.norm(dx, ord=2, axis=2)#[h,w]
            dy = np.linalg.norm(dy, ord=2, axis=2)#[h,w]
            energy_map = dx+dy

            y = std_loss.reshape(-1)
            x = energy_map.reshape(-1)

            ind = np.argsort(x)
            y_r = np.take_along_axis(y, ind, axis=0)
            x_r = np.sort(x)

            x_r = x_r[::10240]
            y_r = y_r[::10240]

            draw(x_r, y_r)
            if batch_idx==5:
                break
    print('finishing handling data!')

def draw(x, y):
    fig = plt.figure()
    name_list = x

    plt.title('img_grad vs loss_std')
    # plt.ylim(ymax=1.5, ymin=-2.2)
    plt.xticks(x, name_list, rotation=90)
    plt.plot(x, y)
    # plt.plot(x, y, label='depth', linewidth=1, color='r', marker='o', markerfacecolor='blue', markersize=5)
    plt.xlabel('img_grad (%)')
    plt.ylabel('loss_std (%)')
    # plt.grid()
    # plt.legend()
    plt.show()
    plt.tight_layout()
    fig.savefig('/node01_data5/monodepth2-test/analysis/table2/img_grad_loss_std.png')
    print('done!')

if __name__ == "__main__":
    opts = parse_args()
    visualize(opts)
