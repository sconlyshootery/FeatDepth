from __future__ import absolute_import, division, print_function
import os
import numpy as np
import argparse
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.datasets.utils import readlines, dump_xyz, compute_ate, transformation_from_parameters
from mono.datasets.kitti_dataset import KITTIOdomDataset
from mono.model.mono_baseline.pose_encoder import PoseEncoder
from mono.model.mono_baseline.pose_decoder import PoseDecoder


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--data_path',
                        default='/node01/odo/dataset')
    parser.add_argument('--output_path',
                        default=None)
    parser.add_argument('--model_path',
                        default='/node01/jobs/io/out/changshu/fm_odo/epoch_10.pth')
    parser.add_argument('--height',
                        default=320)
    parser.add_argument('--width',
                        default=1024)

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


def evaluate(opt):
    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(os.path.join("mono",
                                       "datasets",
                                       "splits",
                                       "odom",
                                       "test_files_{:02d}.txt".format(sequence_id)))

    dataset = KITTIOdomDataset(opt.data_path,
                               filenames,
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

    pred_poses = []

    print("-> Computing pose predictions")
    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in [0, 1]], 1)
            features = pose_encoder(all_color_aug)
            axisangle, translation = pose_decoder(features)
            pred_poses.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
    pred_poses = np.concatenate(pred_poses)

    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate((gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]
    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n  {} Trajectory error: {:0.3f}, std: {:0.3f}\n".format(opts.eval_split, np.mean(ates), np.std(ates)))

    # save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    # np.save(save_path, pred_poses)
    # print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    opts = parse_args()
    opts.eval_split ='odom_9'
    evaluate(opts)
    opts.eval_split = 'odom_10'
    evaluate(opts)
