#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yukun(kunyu@deepmotion.ai)


from __future__ import absolute_import, division, print_function

import os
import cv2
import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append('.')
from data_module.dataset.utils import readlines, dump_xyz, compute_ate, transformation_from_parameters
from data_module.dataset.kitti_dataset import KITTIOdomDataset
from core.model.model_airbus.pose_encoder import PoseEncoder
from core.model.model_airbus.pose_decoder import PoseDecoder
from core.model.model_airbus.pose_model import PoseModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--data_path',
                        default='/home/kun/deepmotion6/Odometry/dataset')
    parser.add_argument('--output_path',
                        default='./evaluation')
    parser.add_argument('--model_path',
                        default='/home/kun/Desktop/epoch_25.pth')
    parser.add_argument('--height',
                        default=192)
    parser.add_argument('--width',
                        default=640)

    parser.add_argument('--seq', default='09')

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
    args = parser.parse_args()
    return args


def save_trajectory(poses, path):

    poses = np.array(poses)
    for i in range(poses.shape[0]):
        poses[i] = np.linalg.inv(np.dot(poses[i], np.linalg.inv(poses[0])))

    poses_flat = [p.flatten()[:-4] for p in poses]

    np.savetxt(path, poses_flat, delimiter=' ')

    return


def evaluate(opt):

   # filenames = readlines('./data_module/dataset/splits/odom/test_files_%s.txt' % opt.seq)

   # dataset = KITTIOdomDataset(opt.data_path,
   #                            filenames,
   #                            opt.height,
   #                            opt.width,
   #                            [0, 1],
   #                            is_train=False,
   #                            img_ext='.png',
   #                            gt_depth_path=None)

    # dataloader = DataLoader(dataset,
    #                        1,
    #                        shuffle=False,
    #                        num_workers=4,
    #                        pin_memory=True,
    #                        drop_last=False)

    model = PoseModel()
    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    model.eval()


    pred_poses = []

    print('running pose %s ...' % (opt.seq))

    print("-> Computing pose predictions")

    imgs = os.listdir(opt.data_path)
    imgs.sort()
    count = 0
    with torch.no_grad():
        for i in range(len(imgs)-1):
            
            img0 = cv2.imread(os.path.join(opt.data_path, imgs[i]))
            img1 = cv2.imread(os.path.join(opt.data_path, imgs[i+1]))
   
   #         all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in [0, 1]], 1)
   #         features = pose_encoder(all_color_aug)
   #         axisangle, translation = pose_decoder(features)
            img0 = transforms.ToTensor()(img0).cuda().unsqueeze(0)
            img1 = transforms.ToTensor()(img1).cuda().unsqueeze(0)
            inputs = torch.cat([img0, img1], 1)

            axisangle, translation = model.forward(inputs)
            pred_poses.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    for i in range(pred_poses.shape[0]):

        if i == 0:
            pose_tem = pred_poses[0]
        else:
            pose_tem = np.matmul(pred_poses[i], pose_tem)

        pred_poses[i] = pose_tem

    save_trajectory(pred_poses, os.path.join(opt.output_path, 'pred.txt'))


if __name__ == "__main__":
    opts = parse_args()
    evaluate(opts)
