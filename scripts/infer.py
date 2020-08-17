from __future__ import absolute_import, division, print_function
import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines
from mono.datasets.kitti_dataset import KITTIRAWDataset

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
CFG_PATH = './config/cfg_kitti_fm.py'
MODEL_PATH = '/node01/jobs/io/out/changshu/wow_refine2/epoch_60.pth'
OUT_PATH = '/node01_data5/monodepth2-test/kitti_test/fm_depth_refine2'
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)


def evaluate():
    filenames = readlines("mono/datasets/splits/kitti_test/val_files.txt")
    cfg = Config.fromfile(CFG_PATH)

    dataset = KITTIRAWDataset(cfg.data['in_path'],
                              filenames,
                              cfg.data['height'],
                              cfg.data['width'],
                              [0],
                              is_train=False,
                              gt_depth_path='/node01_data5/monodepth2-test/monodepth2/gt_depths.npz')

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    cfg.model['imgs_per_gpu'] = 1
    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            outputs = model(inputs)

            img_path = os.path.join(OUT_PATH, 'img_{:0>4d}.jpg'.format(batch_idx))
            plt.imsave(img_path, inputs[("color", 0, 0)][0].squeeze().transpose(0,1).transpose(1,2).cpu().numpy())

            disp = outputs[("disp", 0, 0)]
            pred_disp, _ = disp_to_depth(disp, 0.1, 100)
            pred_disp = pred_disp[0, 0].cpu().numpy()
            pred_disp = cv2.resize(pred_disp, (cfg.data['width'], cfg.data['height']))

            img_path = os.path.join(OUT_PATH, 'disp_{:0>4d}.jpg'.format(batch_idx))
            vmax = np.percentile(pred_disp, 95)
            plt.imsave(img_path, pred_disp, cmap='magma', vmax=vmax)

    print("\n-> Done!")


if __name__ == "__main__":
    evaluate()