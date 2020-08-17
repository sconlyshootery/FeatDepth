#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

cfg = dict(
    model_name = 'MonoDeploy',
    depth_num_layers = 18,
    pose_num_layers = 18,

    depth_pretrained_path = None,
    pose_pretrained_path =  None,
    frame_ids = [],
    imgs_per_gpu = 2,
    scales = [0, 1, 2, 3],

    width = 800,
    height = 256,

    scale = 33,
    model_path = './ckpt/epoch_24.pth',
    cam_params_name = 'kitti',

    cam_params = dict(
        dishuihu_cam_params = dict(
            # image_shape = [1920, 960],
            # intrinsics = [1057.46, 1057.11, 1002.12, 427.939],
            image_shape = [1920, 640],
            intrinsics = [1057.46, 1057.11, 1002.12, 107],
            distortions = [-0.046542, 0.053209, 0.000624, 0.000080],
        ),
        wupeng_cam_params = dict(
            image_shape = [1920, 1208],
            intrinsics = [1810.62316401,  1807.76816575,   940.74346012,   572.52737974],
            distortions = [-0.30987874,  0.14889359, -0.00155969,  0.00224525],
        )
    )
)
