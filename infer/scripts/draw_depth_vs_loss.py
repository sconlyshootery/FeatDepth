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

    interval = 10
    k_list = []
    for k in range(-interval, 1 + interval):
        losses = []
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dataloader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()

                outputs, loss_dict = model(inputs, k/100+1)
                loss = outputs[('min_reconstruct_loss', 0)].cpu().numpy()#[1,h,w]
                print(loss.shape)
                losses.append(loss)
                if batch_idx == 10:
                    break
            losses = np.concatenate(losses, axis=0) #[b,h,w]
            m = losses.mean()
            k_list.append(m)
    print('finishing handling data!')

    y0 = k_list[interval]
    fig = plt.figure()
    x = range(-interval, interval+1)
    y = [(k-y0)/y0*100 for k in k_list]
    name_list = x

    plt.title('loss vs depth')
    plt.ylim(ymax=2, ymin=-2)
    plt.xticks(x, name_list, rotation=90)
    plt.plot(x, y, label='depth', linewidth=1, color='r', marker='o', markerfacecolor='blue', markersize=5)
    plt.xlabel('depth (%)')
    plt.ylabel('loss (%)')
    # plt.grid()
    plt.legend()
    plt.show()
    plt.tight_layout()
    fig.savefig('/node01_data5/monodepth2-test/analysis/table1/loss_depth.png')
    print('done!')

if __name__ == "__main__":
    opts = parse_args()
    visualize(opts)
