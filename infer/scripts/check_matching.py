from __future__ import absolute_import, division, print_function

import os
import cv2
import argparse
import numpy as np
from mmcv import Config
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.datasets.kitti_dataset import KITTIRAWDataset
from mono.model.registry import MONO

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
    parser.add_argument('--data_path',
                        default='/node01_data5/kitti_raw')
    parser.add_argument('--output_path',
                        default='/node01_data5/monodepth2-test/analysis/fig1')

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

    parser.add_argument('--height',
                        default=256)
    parser.add_argument('--width',
                        default=800)
    args = parser.parse_args()
    return args


def visualize(opt,x,y,i):
    device = [0]
    opt.gpu_num = len(device)

    filenames = ['2011_09_26/2011_09_26_drive_0046_sync 0000000085 l']

    dataset = KITTIRAWDataset(opt.data_path,
                              filenames,
                              opt.height,
                              opt.width,
                              [0, -1, 1],
                              is_train=False,
                              gt_depth_path='/node01_data5/monodepth2-test/monodepth2/gt_depths.npz')

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    cfg = Config.fromfile('./config/cfg_kitti_uncert.py')
    cfg.model['imgs_per_gpu'] = 1
    model_name = cfg.model['name']
    model = MONO.module_dict[model_name](cfg.model)

    checkpoint = torch.load(opt.model_path)

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            outputs, loss_dict = model(inputs, 1)
            tgt = inputs[("color", 0, 0)].squeeze().transpose(0,1).transpose(1,2).cpu().numpy()
            src = inputs[("color", 1, 0)].squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()
            pix_coord = outputs[("sample", 1, 0)].squeeze().cpu().numpy()

            xx = (pix_coord[y, x, 0] + 1) / 2 * opt.width
            yy = (pix_coord[y, x, 1] + 1) / 2 * opt.height
            # xx = int(xx)
            # yy = int(yy)
            print('epoch_{}:x={},y={}'.format(i,xx,yy))
            # saveimage(os.path.join(opt.output_path, str(x)+'_'+str(y)+'_'+str(i)+"_tgt.jpg"), tgt, x, y)
            # saveimage(os.path.join(opt.output_path, str(x)+'_'+str(y)+'_'+str(i)+"_src.jpg"), src, xx, yy)



def saveimage(imgname,img,x,y):
    fig = plt.figure()
    plt.axis('off')
    vmax = np.percentile(img, 95)
    vmin = np.percentile(img, 5)
    plt.imshow(img, cmap='jet', vmax=vmax, vmin=vmin)
    plt.plot(x, y, 'ro', linewidth=0.1)
    fig.savefig(imgname, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    opts = parse_args()
    x, y = 340, 150
    for i in range(1,11):
        opts.model_path='/node01_data5/monodepth2-test/model/wow_demo/epoch_'+str(i)+'.pth'
        visualize(opts,x,y,i)
    print('done!')