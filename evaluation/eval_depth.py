#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yukun(kunyu@deepmotion.ai)


from __future__ import absolute_import, division, print_function
import os
import cv2
import sys
import math
import numpy as np
from PIL import Image
from mmcv import Config
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append('.')
from core.model.registry import MONO
from core.model.model_airbus.layers import disp_to_depth
from data_module.dataset.utils import readlines, compute_errors
from data_module.dataset.kitti_dataset import KITTIRAWDataset

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
STEREO_SCALE_FACTOR = 36.0
MIN_DEPTH = 1e-3
MAX_DEPTH = 100


def depth_err_image(depth_est, depth_gt):
    """
    Calculate the error map between disparity estimation and disparity ground-truth
    hot color -> big error, cold color -> small error
    Inputs:
        disp_est: numpy array, disparity estimation map in (Height, Width) layout, range [0,255]
        disp_gt:  numpy array, disparity ground-truth map in (Height, Width) layout, range [0,255]
    Outputs:
        disp_err: numpy array, disparity error map in (Height, Width, 3) layout, range [0,255]
    """
    depth_shape = depth_gt.shape

    # error color map with interval (0, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, inf)/3.0
    # different interval corresponds to different 3-channel projection
    cols = np.array([
        [12.0, float("inf"), 49, 54, 149],
        [8.0, 12.0, 69, 117, 180],
        [5.0, 8.0, 116, 173, 209],
        [3.0, 5.0, 171, 217, 233],
        [2.0, 3.0, 224, 243, 248],
        [1.2, 2.0, 254, 224, 144],
        [0.9, 1.2, 253, 174, 97],
        [0.6, 0.9, 244, 109, 67],
        [0.3, 0.6, 215, 48, 39],
        [0.0, 0.3, 165, 0, 38]
    ])

    E = np.abs(depth_est - depth_gt)

    # based on error color map, project the E within [cols[i,0], cols[i,1]] into 3-channel color image
    depth_err = np.zeros((depth_shape[0], depth_shape[1], 3), dtype='uint8')
    for c_i in range(cols.shape[0]):
        for i in range(depth_shape[0]):
            for j in range(depth_shape[1]):
                if depth_gt[i, j] != 0 and E[i, j] >= cols[c_i, 0] and E[i, j] <= cols[c_i, 1]:
                    depth_err[i, j, 0] = int(cols[c_i, 2])
                    depth_err[i, j, 1] = int(cols[c_i, 3])
                    depth_err[i, j, 2] = int(cols[c_i, 4])

    return depth_err


def log10(x):

    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x.float()) / math.log(10)


def evaluate(pred_disp, gt_depth):

    gt_height = gt_depth.shape[0]
    gt_width = gt_depth.shape[1]
    pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
    pred_depth = 1 / pred_disp

    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
    pred_depth_m = pred_depth[mask]
    gt_depth_m = gt_depth[mask]
    ratio = np.median(gt_depth_m) / np.median(pred_depth_m)
    pred_depth *= ratio
    pred_depth_m *= ratio

    metric = compute_errors(gt_depth_m, pred_depth_m)

    err_image = depth_err_image(pred_depth, gt_depth)

    return metric, err_image


def main(cfg, image_path, gt_path, model_path):

    resize = transforms.Resize((256, 800), interpolation=Image.ANTIALIAS)
    to_tensor = transforms.ToTensor()

    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda()
    model.eval()

    imgs = os.listdir(image_path)
    metrics = []
    print('running depth evaluation...')

    if not os.path.exists('./evaluation/depth_eval'):
        os.mkdir('./evaluation/depth_eval')

    for img in imgs:

        gt = os.path.join(gt_path, img)
        img_save_err = os.path.join('./evaluation/depth_eval', img[:-4]+'_err.jpg')
        img_save_depth = os.path.join('./evaluation/depth_eval', img[:-4]+'_depth.jpg')

        img = os.path.join(image_path, img)
        img = Image.open(img).convert('RGB')
        img = resize(img)
        img = to_tensor(img).unsqueeze(0).cuda()

        gt_depth = cv2.imread(gt, -1) / 255.0

        with torch.no_grad():
            output = model.DepthEncoder(img)
            output = model.DepthDecoder(output)
            disp = output[("disp", 0, 0)]

            pred_disp, depth = disp_to_depth(disp, 0.1, 100)
            pred_disp = pred_disp.cpu()[:, 0].numpy()[0]

        metric, err_image = evaluate(pred_disp, gt_depth)
        metrics.append(metric)

        cv2.imwrite(img_save_err, err_image)
        vmax = np.percentile(pred_disp, 95)
        plt.imsave(img_save_depth, pred_disp, cmap='magma', vmax=vmax)

    mean_errors = np.array(metrics).mean(0)

    print("\n" + ("{:>}| " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{:.3f} " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='KITTI Evaluation toolkit')
    parser.add_argument('--weight_path', type=str,
                        default=None,
                        help='model weight path')

    parser.add_argument('--cfg_path', type=str,
                        default=None,
                        help='config of model')

    parser.add_argument('--gt_path', type=str, default=None,
                        help='ground truth depth path')

    parser.add_argument('--img_path', type=str, default=None,
                        help='image path')

    args = parser.parse_args()

    IMAGE_PATH = args.img_path
    GT_PATH = args.gt_path
    MODEL_PATH = args.weight_path
    CFG_PATH = args.cfg_path

    cfg = Config.fromfile(CFG_PATH)

    main(cfg, IMAGE_PATH, GT_PATH, MODEL_PATH)

