#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

import numpy as np
import torch
import cv2
import time

from mono.model.registry import MONO
from mmcv.runner import load_checkpoint

class MonoInfer(object):
    def __init__(self, cfg):
        """For now, the mono only support mono2 style depth, Neural RGB-> D
        style is in coming
        :param cfg:
        """

        model_name = cfg['model_name']
        model = MONO.module_dict[model_name](cfg)
        self.input_shape = (cfg['height'], cfg['width'])
        self.scale = cfg['scale']

        load_checkpoint(model, cfg['model_path'], map_location='cpu')
        self.model = model.cuda().eval()

    def __transform(self, cv2_img):
        im_tensor = torch.from_numpy(cv2_img.astype(np.float32)).cuda().unsqueeze(0)
        im_tensor = im_tensor.permute(0, 3, 1, 2).contiguous()
        im_tensor = torch.nn.functional.interpolate(im_tensor, self.input_shape,
                                                    mode='bilinear', align_corners=False)
        im_tensor /= 255
        return im_tensor

    def predict(self, cv2_img):
        """
        :param img: str or numpy array
        :return:
        """
        original_height, original_width = cv2_img.shape[:2]
        im_tensor = self.__transform(cv2_img)

        with torch.no_grad():
            input = {}
            input['color_aug', 0, 0] = im_tensor
            outputs = self.model(input)

        disp = outputs[("disp", 0, 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width),
                                                       mode="bilinear", align_corners=False)

        depth = 1/(disp_resized.squeeze().cpu().numpy()*10 + 0.01) * self.scale
        return depth

    def get_img_points(self, img_path, calib):
        cv2_img = cv2.imread(img_path)

        if calib.distortions is not None:
            cv2_img = calib.undistort_img({("color_aug", 0, 0): cv2_img})

        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        torch.cuda.synchronize()
        s_t = time.time()
        depth = self.predict(cv2_img)
        torch.cuda.synchronize()
        print('inference time is ', time.time() - s_t)

        assert depth.shape == cv2_img.shape[:2]
        point_with_color, point_index = self.convert_depth_points(cv2_img, depth, calib)

        return point_with_color, point_index

    def convert_depth_points(self, cv2_img, depth, calib):

        u_index, v_index = np.meshgrid(range(0, depth.shape[1]), range(0, depth.shape[0]))
        colors = cv2_img[v_index, u_index]
        u_index = u_index.reshape((-1, 1))
        v_index = v_index.reshape((-1, 1))
        point_index = np.concatenate((u_index.reshape((-1, 1)), v_index.reshape((-1, 1))), axis=1)
        depth = depth.reshape((-1, 1))
        uv_depth = np.concatenate((u_index, v_index, depth), axis=1)
        pts = calib.project_image_to_rect_p2(uv_depth)
        point_with_color = np.hstack((pts, (colors/255.).reshape((-1, 3))))

        return point_with_color, point_index
