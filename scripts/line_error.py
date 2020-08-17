#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

import os
import numpy as np
import cv2

from mono.mono_infer import MonoInfer

draw_rects = []
errors = []


def get_closest_point(point, target_index, src_index):
    """
    :param point: Nx3
    :param target_index: Nx2
    :param src_index: 2
    :return:
    """
    diff = target_index - src_index
    diff_norm = np.linalg.norm(diff, axis=1)
    index = np.argmin(diff_norm)
    return point[index], target_index.astype(np.int32)[index]

def get_cloest_point_depth(point, src_index):
    index = src_index[1]*img_shape[1] + src_index[0]
    return point[index]

def convert_point_to_bev(point):
    v = int(img_shape[0] - (point[2]/100) * img_shape[0])
    u = int((point[0] + 20)*10) + img_shape[1]
    return (u, v)

# mouse callback function
def draw_line(event,x,y,flags,param):
    global draw_rects
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x y is ', (x, y))
        if draw_rects.__len__() < 2:
            draw_rects.append((x, y))

    if draw_rects.__len__() == 2:
        cv2.line(img, draw_rects[0], draw_rects[1], (255, 0, 0), thickness=2)
        point_s, s_index = get_closest_point(pts_rect, pts_index, draw_rects[0])
        point_e, e_index = get_closest_point(pts_rect, pts_index, draw_rects[1])
        s_bev = convert_point_to_bev(point_s)
        e_bev = convert_point_to_bev(point_e)
        cv2.line(img, s_bev, e_bev, (255, 0, 0), thickness=2)
        print('s_index is ', s_index)
        cv2.circle(img, tuple(s_index), 4, (0, 255, 0), thickness=2)
        cv2.circle(img, tuple(e_index), 4, (0, 255, 0), thickness=2)
        print('points are ', point_s, point_e)

        point_s_d = get_cloest_point_depth(pc, tuple(s_index))
        point_e_d = get_cloest_point_depth(pc, tuple(e_index))
        s_bev = convert_point_to_bev(point_s_d)
        e_bev = convert_point_to_bev(point_e_d)
        cv2.line(img, s_bev, e_bev, (0, 255, 0), thickness=2)
        print('depth points are ', point_s_d, point_e_d)

        errors.append((point_s[2], point_e[2], np.abs(point_s[0] - point_s_d[0]),
                       np.abs(point_e[0] - point_e_d[0])))
        print('errors is ', errors[-1])
        draw_rects = []


if __name__ == '__main__':
    # root_dir = '/home/duan/dm6/2011_09_26/2011_09_26_drive_0027_sync/'
    root_dir = '/home/duan/dm6/2011_09_26/2011_09_26_drive_0029_sync/'
    img_dir = os.path.join(root_dir, 'image_02/data')
    velo_dir = os.path.join(root_dir, 'velodyne_points/data')

    out_path = './result'
    cfg = {
        'model_name': 'mono_stereo',
        'num_layer' : 50,
        'input_shape' : (256, 800),
        # 'input_shape' : (192, 640),
        'scale': 5.4,
        # 'scale': 6.2,
        'model_path': './tmp/mono_stereo.pth'
    }

    infer = MonoInfer(cfg)
    imgs = os.listdir(img_dir)
    os.makedirs('./result', exist_ok=True)
    for img_n in sorted(imgs):
        img_path = os.path.join(img_dir, img_n)
        img = cv2.imread(img_path)
        img_shape = img.shape
        img_bev = np.zeros([img_shape[0], 400, 3], dtype=np.uint8)
        img = np.concatenate((img, img_bev), axis=1)
        cv2.namedWindow('image')

        point_with_color = infer.vis_result(img_path, './result')

        pc = point_with_color[:, :3]

        velo = np.fromfile(os.path.join(velo_dir, img_n.replace('.png', '.bin')), dtype=np.float32)
        velo = velo.reshape((-1, 4))[:, :-1]
        pts_rect = infer.calib.project_velo_to_rect(velo)
        pts_rect, pts_index = infer.calib.get_valid_flag(pts_rect, img_shape[:2])

        cv2.setMouseCallback('image', draw_line)
        while(1):
            cv2.imshow('image', img)
            k_input = cv2.waitKey(20) & 0xFF
            if k_input == 27:
                break
            elif k_input == 100:
                errors.pop()
            elif k_input == 115:
                print('errors is ', errors)
                with open('./line_error.txt', 'a+') as f:
                    for e in errors:
                        f.write('{} {} {} {} \n'.format(e[0], e[1], e[2], e[3]))
                errors = []

        cv2.destroyAllWindows()




