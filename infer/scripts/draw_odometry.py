from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.datasets.euroc_dataset import FolderDataset
from mono.datasets.kitti_dataset import KITTIOdomDataset
from mono.datasets.utils import readlines,transformation_from_parameters
from mono.model.mono_baseline.pose_encoder import PoseEncoder
from mono.model.mono_baseline.pose_decoder import PoseDecoder
from kitti_evaluation_toolkit import kittiOdomEval


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
    parser.add_argument('--model_path',
                        default='/node01/jobs/io/out/changshu/fm_odo/epoch_20.pth',
                        help='model save path')
    parser.add_argument('--data_path',
                        default='/node01/odo/dataset')
    parser.add_argument('--output_path',
                        default='/node01_data5/monodepth2-test/traj')
    parser.add_argument('--height',
                        default=192)
    parser.add_argument('--width',
                        default=640)

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


def odo(opt):
    if opt.kitti:
        filenames = readlines("mono/datasets/splits/odom/test_files_{:02d}.txt".format(opt.sequence_id))

        dataset = KITTIOdomDataset(opt.data_path,
                                   filenames,
                                   opt.height,
                                   opt.width,
                                   [0, 1],
                                   is_train=False,
                                   img_ext='.png',
                                   gt_depth_path=None)
    else:
        dataset = FolderDataset(opt.data_path,
                                None,
                                opt.height,
                                opt.width,
                                [0, 1],
                                is_train=False,
                                img_ext='.png',
                                gt_depth_path=None)

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    pose_encoder = PoseEncoder(18, None, 2)
    pose_decoder = PoseDecoder(pose_encoder.num_ch_enc)

    checkpoint = torch.load(opt.model_path)
    for name, param in pose_encoder.state_dict().items():
        pose_encoder.state_dict()[name].copy_(checkpoint['state_dict']['PoseEncoder.' + name])
    for name, param in pose_decoder.state_dict().items():
        pose_decoder.state_dict()[name].copy_(checkpoint['state_dict']['PoseDecoder.' + name])
    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    global_pose = np.identity(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in [0,1]], 1)
            axisangle, translation = pose_decoder(pose_encoder(all_color_aug))
            g = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
            backward_transform = g.squeeze().cpu().numpy()#the transformation from frame +1 to frame 0
            global_pose = global_pose @ np.linalg.inv(backward_transform)
            poses.append(global_pose[0:3, :].reshape(1, 12))
    poses = np.concatenate(poses, axis=0)

    if opt.kitti:
        filename = os.path.join(opt.output_path, "{:02d}_pred.txt".format(opt.sequence_id))
    else:
        filename = os.path.join(opt.output_path, "fm_ms_euroc_mh04_diff_3.txt")

    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
    if opt.kitti:
        opt.gt_dir = './scripts/gt_pose'
        opt.result_dir = opt.output_path
        opt.eva_seqs = '{:02d}_pred'.format(opt.sequence_id)
        opt.toCameraCoord = False
        pose_eval = kittiOdomEval(opt)
        pose_eval.eval(toCameraCoord=opt.toCameraCoord)  # set the value according to the predicted results
    print('saving into ', filename)


if __name__ == "__main__":
    opts = parse_args()
    opts.kitti = True
    if opts.kitti:
        opts.sequence_id = 9
        odo(opts)
        opts.sequence_id = 10
        odo(opts)
    else:
        odo(opts)
    print("you can also run 'evo_traj kitti -s *.txt *.txt --ref=*.txt -p --plot_mode=xz' in terminal for visualization")