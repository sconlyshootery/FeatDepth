from __future__ import absolute_import, division, print_function

import os
import cv2
import argparse
import numpy as np
from mmcv import Config
import matplotlib.pyplot as plt
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.datasets.get_dataset import get_dataset
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth

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
    parser.add_argument('--output_path',
                        default='/node01_data5/monodepth2-test/analysis/visualize_make3d')
    parser.add_argument('--model_path',
                        default = '/node01/jobs/io/out/changshu/bestmodel/bestmodel.pth')
    parser.add_argument('--cfg_path',
                        default = './config/cfg_make3d_fm.py')

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


def visualize(opt):
    cfg = Config.fromfile(opt.cfg_path)
    dataset = get_dataset(cfg.data, training=False)
    dataloader = DataLoader(dataset,
                            cfg.IMGS_PER_GPU,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    model_name = cfg.model['name']
    model = MONO.module_dict[model_name](cfg.model)

    checkpoint = torch.load(opt.model_path)

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.cuda()
    model.eval()

    with torch.no_grad():
        if not os.path.exists(opt.output_path):
            os.mkdir(opt.output_path)

        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            outputs = model(inputs)
            print(batch_idx)
            batchsize = outputs[("disp", 0, 0)].size(0)
            for i in range(batchsize):
                for scale in [0]:
                # for scale in [0,1,2,3]:
                    # for frame_id in [0,-1,1]:
                    for frame_id in [0]:
                        disp = outputs[("disp", 0, scale)][i].squeeze().cpu().numpy()
                        # depth = disp_to_depth(disp, 0.1, 100)
                        vmax = np.percentile(disp, 95)
                        plt.imsave(os.path.join(opt.output_path,
                                                str(batch_idx * batchsize + i) + "_disp_{}.jpg".format(scale)),
                                                disp,
                                                cmap='magma',
                                                vmax=vmax)

                        img = inputs[('color', frame_id, 0)][i].squeeze().transpose(0,1).transpose(1,2).cpu().numpy()
                        plt.imsave(os.path.join(opt.output_path,
                                                str(batch_idx * batchsize + i) + "_img_{}.jpg".format(frame_id)), img)

                        # if frame_id!=0:
                        #     warp_img = outputs[("color", frame_id, scale)][i].squeeze().clamp(0, 1).transpose(0,1).transpose(1,2).cpu().numpy()
                        #     plt.imsave(os.path.join(opt.output_path, str(batch_idx * batchsize + i) + "_warp_{}_{}.jpg".format(frame_id, scale)), warp_img)

                        # index = outputs[("min_index", scale)][i].mul(100).squeeze().cpu().numpy()
                        # plt.imsave(os.path.join(opt.output_path,
                        #                         str(batch_idx * batchsize + i) + "_index_{}.jpg".format(scale)), index)

                        # forward_transform = outputs[("cam_T_cam", 0, -1)][i].squeeze().cpu().numpy()#the transformation from frame -1 to frame 0
                        # backward_transform = outputs[("cam_T_cam", 0, 1)][i].squeeze().cpu().numpy()#the transformation from frame +1 to frame 0

            # if batch_idx==40:
            #     break
    print('done!')


if __name__ == "__main__":
    opts = parse_args()
    visualize(opts)
