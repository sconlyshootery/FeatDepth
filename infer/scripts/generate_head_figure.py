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
                        default='/node01_data5/monodepth2-test/analysis/head')
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
    parser.add_argument('--batchsize',
                        default=5)
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
            for i in range(opt.batchsize):
                outputs, loss_dict = model(inputs, 1.2)
                max_loss = outputs[('min_reconstruct_loss', 0)].cpu().numpy()#[b,h,w]

                outputs, loss_dict = model(inputs, 1)
                original_loss = outputs[('min_reconstruct_loss', 0)].cpu().numpy()  # [b,h,w]

                disp = outputs[('disp', 0, 0)]  # [b,h,w]
                disp = F.interpolate(disp, [opt.height, opt.width], mode="bilinear", align_corners=False)
                disp = disp.cpu().numpy()
                disp = np.squeeze(disp[i])
                max_loss = np.squeeze(max_loss[i])
                original_loss = np.squeeze(original_loss[i])

                target = inputs[("color", 0, 0)][i].squeeze().transpose(0,1).transpose(1,2).cpu().numpy()

                saveimage(os.path.join(opt.output_path, 'depth', str(batch_idx * opt.batchsize + i) + ".jpg"), disp, 'magma')
                saveimage(os.path.join(opt.output_path, 'img', str(batch_idx * opt.batchsize + i) + ".jpg"), target, None)
                saveimage(os.path.join(opt.output_path, 'ori_loss', str(batch_idx * opt.batchsize + i) + ".jpg"), original_loss, 'jet')
                saveimage(os.path.join(opt.output_path, 'loss', str(batch_idx * opt.batchsize + i) + ".jpg"), max_loss, 'jet')
    print('done!')

def saveimage(imgname,img,cmap):
    vmax = np.percentile(img, 95)
    vmin = np.percentile(img, 5)
    plt.imsave(imgname, img, cmap=cmap, vmax=vmax, vmin=vmin)


if __name__ == "__main__":
    opts = parse_args()
    visualize(opts)
