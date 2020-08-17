#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

import os
import numpy as np

from mono.infer import MonoInfer
from mono.vis import calibration
from mmcv import Config

DEBUG = True
LIDAR = True

if DEBUG:
    from mono.vis.viewer import MapViewer
    map_viewer = MapViewer()

if __name__ == '__main__':
    root_dir = '/home/duan/dm6/rawdata/2011_09_26/2011_09_26_drive_0022_sync/'
    img_dir = os.path.join(root_dir, 'image_02/data')
    velo_dir = os.path.join(root_dir, 'velodyne_points/data')
    out_path = './result'

    cfg = Config.fromfile('./config/inference_cfg.py')['cfg']

    if cfg['cam_params_name'] == 'kitti':
        calib = calibration.KittiCalibration('./mono/vis/kitti_calib.txt')
    else:
        calib_name = cfg['cam_params_name']
        calib_params = cfg['cam_params'][cfg['cam_params_name']]
        calib = calibration.UserCalibration.from_params(
            calib_params['intrinsics'],
            calib_params['distortions']
        )

    infer = MonoInfer(cfg)

    imgs = os.listdir(img_dir)
    os.makedirs('./result', exist_ok=True)
    for img in sorted(imgs):
        print('img is ', img)
        img_path = os.path.join(img_dir, img)
        point_with_color, _ = infer.get_img_points(img_path, calib)

        pc = point_with_color[:, :3]

        if DEBUG:
            if not LIDAR:
                map_viewer.update(tmp_pc=point_with_color)
            else:
                velo = np.fromfile(os.path.join(velo_dir, img.replace('.png', '.bin')), dtype=np.float32)
                velo = velo.reshape((-1, 4))[:, :-1]
                velo_in_rect = calib.project_velo_to_rect(velo)
                # pc.astype(np.float16).tofile(os.path.join(velo_save_dir, img.replace('.png', '.bin')))
                map_viewer.update(tmp_pc=point_with_color, velo=velo_in_rect)




