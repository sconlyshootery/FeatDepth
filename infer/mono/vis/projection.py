#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

import numpy as np

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

class Projector(object):
    def __init__(self, calib):
        """
        :param calib:
        """
        self.calib = calib

    def convert(self, disp, img, degree=30):
        assert disp.shape == img.shape[:2]

        u_index, v_index = np.meshgrid(range(0, disp.shape[1]), range(0, disp.shape[0]))
        depth = self.calib.f_u * self.calib.baseline / disp[v_index, u_index]
        colors = img[v_index, u_index]
        u_index = u_index.reshape((-1, 1))
        v_index = v_index.reshape((-1, 1))
        depth = depth.reshape((-1, 1))
        uv_depth = np.concatenate((u_index, v_index, depth), axis=1)
        pts = self.calib.project_image_to_rect_p2(uv_depth)

        t = np.array([0, 20, 10]).reshape((3,1))
        r = rotx(-1*degree*np.pi/180)

        calib_k = self.calib.camera_k
        new_p = np.dot(calib_k, np.hstack((r, t)))

        pts_3d_rect = self.calib.cart2hom(pts)
        pts_2d = np.dot(pts_3d_rect, np.transpose(new_p)) # nx4 . 4x3 = nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        pts_2d = pts_2d[:, 0:2]
        pts_2d = pts_2d.astype(np.int)


        pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, 1023)
        pts_2d[:, 1] = np.clip(pts_2d[:, 1], 400, 1499)
        new_image = np.zeros((1500-400, 1024, 3)).astype(np.uint8)
        new_image[pts_2d[:, 1]-400, pts_2d[:, 0]] = colors.reshape((-1, 3))
        # pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, 1023)
        # pts_2d[:, 1] = np.clip(pts_2d[:, 1], 1400, 3200)
        # new_image = np.zeros((3200-1400, 1024, 3)).astype(np.uint8)
        # new_image[pts_2d[:, 1]-1400, pts_2d[:, 0]] = colors.reshape((-1, 3))
        return new_image







