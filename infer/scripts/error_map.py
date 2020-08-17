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
from mono.datasets.utils import readlines, batch_post_process_disparity, compute_errors
from mono.datasets.kitti_dataset import KITTIRAWDataset

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH=0.1
MAX_DEPTH=100
PRED_ERROR = True
CFG_PATH = './config/cfg_kitti_baseline_s.py'
MODEL_PATH = '/node01_data5/monodepth2-test/model/s_baseline/s_baseline_50_256_800.pth' #'/node01_data5/monodepth2-test/model/ms_baseline/ms_baseline_50_256_800.pth' #/data1/jobs/io/out/changshu/canet


def analysis(gt, pred):
    # return np.abs(gt-pred)
    factor = 0.1
    return np.logical_or(((1-factor)*gt > pred), (pred > gt*(1+factor)))


def saveimage(imgname,img,x=None,y=None):
    fig = plt.figure()
    plt.axis('off')
    vmax = np.percentile(img, 95)
    vmin = np.percentile(img, 5)
    plt.imshow(img, cmap='jet', vmax=vmax, vmin=vmin)
    if x is not None and y is not None:
        plt.plot(x, y, 'ro', linewidth=0.1)
    fig.savefig(imgname, bbox_inches='tight')
    plt.close(fig)


def evaluate():
    filenames = readlines("mono/datasets/splits/exp/val_files.txt")
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

    pred_disps = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            img = inputs[("color", 0, 0)].squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()
            saveimage(os.path.join(os.path.dirname(cfg.data['in_path']), 'monodepth2-test', 'error', str(batch_idx) + '_img.jpg'), img)

            outputs = model(inputs)
            disp = outputs[("disp", 0, 0)]
            pred_disp, _ = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)
    pred_disps = np.concatenate(pred_disps)

    gt_path = os.path.join(os.path.dirname(cfg.data['in_path']), 'monodepth2-test', 'monodepth2', "gt_depths.npz")
    gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")
    if cfg.data['stereo_scale']:
        print('using baseline')
    else:
        print('using mean scaling')

    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))

        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth_org = pred_depth
        gt_depth_org = gt_depth

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)

        if cfg.data['stereo_scale']:
            ratio = STEREO_SCALE_FACTOR

        pred_depth *= ratio
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))

        if PRED_ERROR:
            pred_depth_org *= ratio
            pred_depth_org[pred_depth_org < MIN_DEPTH] = MIN_DEPTH
            pred_depth_org[pred_depth_org > MAX_DEPTH] = MAX_DEPTH
            pred_depth_org = pred_depth_org * mask
            gt_depth_org = gt_depth_org * mask

            error_map = analysis(gt_depth_org, pred_depth_org)
            max = np.max(error_map)
            error_map = (error_map / max * 255).astype(np.uint8)
            error_map = cv2.resize(error_map, (cfg.data['width'], cfg.data['height']), interpolation=cv2.INTER_NEAREST)
            # saveimage(os.path.join(os.path.dirname(cfg.data['in_path']), 'monodepth2-test', 'error', str(i) + '_errormap.jpg'), error_map)
            cv2.imwrite(os.path.join(os.path.dirname(cfg.data['in_path']), 'monodepth2-test', 'error', str(i) + '_errormap.jpg'), error_map)

            gt_map = (mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(os.path.dirname(cfg.data['in_path']), 'monodepth2-test', 'error', str(i) + '_gtmap.jpg'), gt_map)

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    mean_errors = np.array(errors).mean(0)
    print("\n" + ("{:>}| " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{:.3f} " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    evaluate()