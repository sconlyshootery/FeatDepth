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
                            opt.batchsize,
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
                losses = []
                interval = 10
                for k in range(-interval, 1+interval):
                    outputs, loss_dict = model(inputs, k/100+1)
                    loss = outputs[('min_reconstruct_loss', 0)].cpu().numpy()#[b,h,w]
                    loss = loss[:, np.newaxis, :,:]
                    losses.append(loss)

                outputs, loss_dict = model(inputs, 1)
                pred = outputs[('min_reconstruct_loss', 0)].cpu().numpy()  # [b,h,w]

                disp = outputs[('disp', 0, 0)]  # [b,h,w]
                disp = F.interpolate(disp, [opt.height, opt.width], mode="bilinear", align_corners=False)
                disp = disp.cpu().numpy()

                losses = np.concatenate(losses, axis=1)
                mean_loss = np.mean(losses, axis=1)
                std_loss = np.std(losses, axis=1)
                max_loss = np.amax(losses, axis=1)
                min_loss = np.amin(losses, axis=1)

                max_min_loss = max_loss-min_loss
                pred_min = pred - min_loss

                pred_min = np.squeeze(pred_min[i])
                max_min = np.squeeze(max_min_loss[i])
                mean = np.squeeze(mean_loss[i])
                std = np.squeeze(std_loss[i])
                pred_loss = np.squeeze(pred[i])
                disp = np.squeeze(disp[i])

                target = inputs[("color", 0, 0)][i].squeeze().transpose(0,1).transpose(1,2).cpu().numpy()
                forward = inputs[("color", -1, 0)][i].squeeze().transpose(0,1).transpose(1,2).cpu().numpy()
                backward = inputs[("color", 1, 0)][i].squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()

                saveimage(os.path.join(opt.output_path, 'disp', str(batch_idx * opt.batchsize + i) + ".jpg"), disp)
                saveimage(os.path.join(opt.output_path, 'mean', str(batch_idx * opt.batchsize + i) + ".jpg"), mean)
                saveimage(os.path.join(opt.output_path, 'std', str(batch_idx * opt.batchsize + i) + ".jpg"), std)
                saveimage(os.path.join(opt.output_path, 'img', str(batch_idx * opt.batchsize + i) + ".jpg"), target)
                saveimage(os.path.join(opt.output_path, 'forward', str(batch_idx * opt.batchsize + i) + ".jpg"), forward)
                saveimage(os.path.join(opt.output_path, 'backward', str(batch_idx * opt.batchsize + i) + ".jpg"), backward)
                saveimage(os.path.join(opt.output_path, 'maxmin', str(batch_idx * opt.batchsize + i) + ".jpg"), max_min)
                saveimage(os.path.join(opt.output_path, 'predmin', str(batch_idx * opt.batchsize + i) + ".jpg"), pred_min)
                saveimage(os.path.join(opt.output_path, 'predloss', str(batch_idx * opt.batchsize + i) + ".jpg"), pred_loss)

            if batch_idx==5:
                break
    print('done!')

def saveimage(imgname,img):
    vmax = np.percentile(img, 95)
    vmin = np.percentile(img, 5)
    plt.imsave(imgname, img, cmap='jet', vmax=vmax, vmin=vmin)


if __name__ == "__main__":
    opts = parse_args()
    visualize(opts)
